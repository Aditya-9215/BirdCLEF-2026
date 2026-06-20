"""
BirdCLEF+ 2026 PyTorch ResNet baseline for Kaggle Notebooks.

This script expects the competition dataset to be attached to the Kaggle
Notebook. It does not download the dataset locally. On Kaggle, the data should
be available somewhere under /kaggle/input, commonly:

    /kaggle/input/birdclef-2026

The pipeline:
1. Finds train.csv, taxonomy.csv, sample_submission.csv, and audio folders.
2. Trains a ResNet-34 on 5-second mel spectrograms.
3. Uses primary labels, secondary labels, and optionally labeled soundscape
   segments when train_soundscapes_labels.csv is present.
4. Writes submission.csv for the hidden test_soundscapes directory.
"""

from __future__ import annotations

import ast
import gc
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34


COMPETITION_URL = "https://www.kaggle.com/competitions/birdclef-2026"
COMPETITION_SLUG = "birdclef-2026"


@dataclass
class CFG:
    seed: int = 42
    sample_rate: int = 32_000
    duration: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 16_000
    model_name: str = "resnet34"
    epochs: int = 2
    batch_size: int = 32
    valid_batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    valid_size: float = 0.15
    max_train_rows: Optional[int] = None
    include_labeled_soundscapes: bool = True
    checkpoint_name: str = "birdclef2026_resnet34_baseline.pt"
    submission_name: str = "submission.csv"

    @property
    def audio_samples(self) -> int:
        return int(self.sample_rate * self.duration)


cfg = CFG()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_data_dir() -> Path:
    """Find the Kaggle-mounted competition input folder."""
    candidates = [
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("/kaggle/input/birdclef-plus-2026"),
        Path("/kaggle/input/birdclef-2026-data"),
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "sample_submission.csv").exists():
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in input_root.glob("*"):
            if (candidate / "train.csv").exists() and (candidate / "sample_submission.csv").exists():
                return candidate

    raise FileNotFoundError(
        "Could not find BirdCLEF+ 2026 data. Attach the Kaggle competition "
        f"dataset ({COMPETITION_URL}) to the notebook and rerun."
    )


def load_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_csv = pd.read_csv(data_dir / "train.csv")
    taxonomy_csv = pd.read_csv(data_dir / "taxonomy.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    print(f"Data dir: {data_dir}")
    print(f"train.csv: {train_csv.shape}")
    print(f"taxonomy.csv: {taxonomy_csv.shape}")
    print(f"sample_submission.csv: {sample_submission.shape}")
    return train_csv, taxonomy_csv, sample_submission


def parse_secondary_labels(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def parse_semicolon_labels(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def make_records(
    data_dir: Path,
    train_csv: pd.DataFrame,
    class_names: Sequence[str],
    include_labeled_soundscapes: bool = True,
) -> pd.DataFrame:
    class_set = set(class_names)
    records: List[Dict[str, object]] = []

    for row in train_csv.itertuples(index=False):
        primary = str(getattr(row, "primary_label"))
        labels = [primary]
        labels.extend(parse_secondary_labels(getattr(row, "secondary_labels", [])))
        labels = sorted({label for label in labels if label in class_set})
        if not labels:
            continue
        filename = str(getattr(row, "filename"))
        records.append(
            {
                "source": "train_audio",
                "path": str(data_dir / "train_audio" / filename),
                "primary_label": primary,
                "labels": labels,
                "start": np.nan,
                "end": np.nan,
            }
        )

    labels_path = data_dir / "train_soundscapes_labels.csv"
    soundscape_dir = data_dir / "train_soundscapes"
    if include_labeled_soundscapes and labels_path.exists() and soundscape_dir.exists():
        soundscape_labels = pd.read_csv(labels_path)
        for row in soundscape_labels.itertuples(index=False):
            labels = parse_semicolon_labels(getattr(row, "primary_label"))
            labels = sorted({label for label in labels if label in class_set})
            if not labels:
                continue
            records.append(
                {
                    "source": "train_soundscape",
                    "path": str(soundscape_dir / str(getattr(row, "filename"))),
                    "primary_label": labels[0],
                    "labels": labels,
                    "start": float(getattr(row, "start")),
                    "end": float(getattr(row, "end")),
                }
            )

    records_df = pd.DataFrame(records)
    records_df = records_df[records_df["path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"Training records after file check: {records_df.shape}")
    print(records_df["source"].value_counts(dropna=False))
    return records_df


def split_records(records_df: pd.DataFrame, valid_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if records_df.empty:
        raise ValueError("No training records were found.")

    counts = records_df["primary_label"].value_counts()
    can_stratify = records_df["primary_label"].map(counts).min() >= 2
    stratify = records_df["primary_label"] if can_stratify else None
    train_df, valid_df = train_test_split(
        records_df,
        test_size=valid_size,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def load_audio(path: str, sr: int, offset: Optional[float] = None, duration: Optional[float] = None) -> np.ndarray:
    offset_value = 0.0 if offset is None or np.isnan(offset) else float(offset)
    kwargs = {"sr": sr, "mono": True}
    if offset_value > 0:
        kwargs["offset"] = offset_value
    if duration is not None and not np.isnan(duration):
        kwargs["duration"] = float(duration)
    audio, _ = librosa.load(path, **kwargs)
    return audio.astype(np.float32, copy=False)


def crop_or_pad(audio: np.ndarray, target_samples: int, mode: str) -> np.ndarray:
    if len(audio) < target_samples:
        pad = target_samples - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    elif len(audio) > target_samples:
        if mode == "train":
            start = random.randint(0, len(audio) - target_samples)
        else:
            start = (len(audio) - target_samples) // 2
        audio = audio[start : start + target_samples]
    return audio


def audio_to_mel(audio: np.ndarray, cfg: CFG) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)


class BirdCLEFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], cfg: CFG, mode: str):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.cfg = cfg
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        duration = None
        if row["source"] == "train_soundscape":
            duration = float(row["end"]) - float(row["start"])
        audio = load_audio(
            str(row["path"]),
            sr=self.cfg.sample_rate,
            offset=row["start"] if row["source"] == "train_soundscape" else None,
            duration=duration,
        )
        audio = crop_or_pad(audio, self.cfg.audio_samples, self.mode)
        image = audio_to_mel(audio, self.cfg)

        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in row["labels"]:
            label = str(label)
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        return image, target


def build_model(num_classes: int) -> nn.Module:
    try:
        model = resnet34(weights=None)
    except TypeError:
        model = resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(
        1,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def fit_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    cfg: CFG,
    device: torch.device,
) -> nn.Module:
    train_dataset = BirdCLEFDataset(train_df, class_to_idx, cfg, mode="train")
    valid_dataset = BirdCLEFDataset(valid_df, class_to_idx, cfg, mode="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(num_classes=len(class_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        valid_loss = validate(model, valid_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} time={elapsed:.1f}s"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, cfg.checkpoint_name)
            print(f"Saved checkpoint: {cfg.checkpoint_name}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def segment_to_mel(full_audio: np.ndarray, end_time: int, cfg: CFG) -> torch.Tensor:
    end_sample = int(end_time * cfg.sample_rate)
    start_sample = max(0, end_sample - cfg.audio_samples)
    segment = full_audio[start_sample:end_sample]
    segment = crop_or_pad(segment, cfg.audio_samples, mode="valid")
    return audio_to_mel(segment, cfg)


def parse_row_id(row_id: str) -> Tuple[str, int]:
    stem, end_time = row_id.rsplit("_", 1)
    return stem, int(end_time)


@torch.no_grad()
def predict_test(
    model: nn.Module,
    data_dir: Path,
    sample_submission: pd.DataFrame,
    class_names: Sequence[str],
    cfg: CFG,
    device: torch.device,
) -> pd.DataFrame:
    test_dir = data_dir / "test_soundscapes"
    submission = sample_submission.copy()
    submission[class_names] = 0.0

    if not test_dir.exists():
        print("test_soundscapes directory was not found. Writing a zero-filled submission template.")
        return submission

    test_files = sorted(test_dir.glob("*.ogg"))
    if not test_files:
        print("No hidden test files are visible in this session. Writing a zero-filled submission template.")
        return submission

    row_map: Dict[str, List[Tuple[int, int]]] = {}
    for row_idx, row_id in enumerate(submission["row_id"].astype(str).tolist()):
        stem, end_time = parse_row_id(row_id)
        row_map.setdefault(stem, []).append((row_idx, end_time))

    model.eval()
    file_by_stem = {path.stem: path for path in test_files}
    batch_images: List[torch.Tensor] = []
    batch_indices: List[int] = []

    def flush_batch() -> None:
        nonlocal batch_images, batch_indices, submission
        if not batch_images:
            return
        images = torch.stack(batch_images).to(device)
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()
        submission.iloc[batch_indices, submission.columns.get_indexer(class_names)] = probs
        batch_images = []
        batch_indices = []

    for stem, rows in row_map.items():
        path = file_by_stem.get(stem)
        if path is None:
            continue
        full_audio = load_audio(str(path), sr=cfg.sample_rate)
        full_audio = crop_or_pad(full_audio, int(60 * cfg.sample_rate), mode="valid")
        for row_idx, end_time in sorted(rows, key=lambda item: item[1]):
            batch_images.append(segment_to_mel(full_audio, end_time, cfg))
            batch_indices.append(row_idx)
            if len(batch_images) >= cfg.valid_batch_size:
                flush_batch()
        del full_audio
        gc.collect()

    flush_batch()
    return submission


def main() -> None:
    seed_everything(cfg.seed)
    data_dir = resolve_data_dir()
    train_csv, taxonomy_csv, sample_submission = load_tables(data_dir)

    class_names = [column for column in sample_submission.columns if column != "row_id"]
    if not class_names:
        class_names = taxonomy_csv["primary_label"].astype(str).tolist()
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    print(f"Classes: {len(class_names)}")

    records_df = make_records(
        data_dir=data_dir,
        train_csv=train_csv,
        class_names=class_names,
        include_labeled_soundscapes=cfg.include_labeled_soundscapes,
    )
    if cfg.max_train_rows is not None and len(records_df) > cfg.max_train_rows:
        records_df = records_df.sample(cfg.max_train_rows, random_state=cfg.seed).reset_index(drop=True)
        print(f"Using cfg.max_train_rows={cfg.max_train_rows}: {records_df.shape}")

    train_df, valid_df = split_records(records_df, valid_size=cfg.valid_size, seed=cfg.seed)
    print(f"Train rows: {len(train_df)} | Valid rows: {len(valid_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = fit_model(train_df, valid_df, class_to_idx, cfg, device)

    submission = predict_test(model, data_dir, sample_submission, class_names, cfg, device)
    submission.to_csv(cfg.submission_name, index=False)
    print(f"Wrote {cfg.submission_name}: {submission.shape}")


if __name__ == "__main__":
    main()
