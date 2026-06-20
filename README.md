# BirdCLEF+ 2026 ResNet Baseline

Competition: <https://www.kaggle.com/competitions/birdclef-2026>

This project is a simple working PyTorch baseline for the Kaggle BirdCLEF+ 2026 competition. It is designed to run inside a Kaggle Notebook with the competition dataset attached under `/kaggle/input`. It does not download or store the 16 GB dataset locally.

## Competition Notes

From the Data tab:

- The task is multi-species audio detection for Brazilian Pantanal recordings.
- Training audio is in `train_audio/`, short OGG files resampled to 32 kHz.
- Hidden test audio appears only during notebook scoring in `test_soundscapes/`.
- Test soundscapes are 1-minute OGG files at 32 kHz.
- Predictions are required for 5-second segments.
- `sample_submission.csv` contains `row_id` plus 234 species columns.
- `row_id` is `[soundscape_filename]_[end_time]`, for example `BC2026_Test_0001_S05_20250227_010002_20`.
- `taxonomy.csv` lists the 234 target classes.
- `train_soundscapes_labels.csv`, when present, gives expert labels for some 5-second training soundscape segments.

## Baseline Approach

The program in `birdclef_2026_resnet_baseline.py`:

- Finds the mounted Kaggle dataset automatically under `/kaggle/input`.
- Reads `train.csv`, `taxonomy.csv`, and `sample_submission.csv`.
- Builds 5-second mel spectrograms from 32 kHz OGG audio.
- Uses `torchvision.models.resnet34` as a CNN over spectrogram images.
- Replaces the first ResNet convolution with a 1-channel input layer.
- Replaces the final ResNet classifier with a 234-output head.
- Trains with `BCEWithLogitsLoss` for multi-label probabilities.
- Uses primary labels plus parsed secondary labels from `train.csv`.
- Optionally adds labels from `train_soundscapes_labels.csv`.
- Writes `submission.csv` in the Kaggle working directory.

## Project Files

- `birdclef_2026_resnet_baseline.ipynb` - self-contained Kaggle Notebook version.
- `birdclef_2026_resnet_baseline.py` - same baseline as a Python script.
- `README.md` - competition notes and run instructions.

## How To Run On Kaggle

1. Open a Kaggle Notebook for the BirdCLEF+ 2026 competition.
2. Attach the competition dataset with Kaggle's **Add data** panel.
3. Upload and run `birdclef_2026_resnet_baseline.ipynb`.

Alternatively, copy or upload `birdclef_2026_resnet_baseline.py` into the notebook and run:

```python
%run /kaggle/working/birdclef_2026_resnet_baseline.py
```

If you paste the script into a notebook cell instead, run the cell directly.

The script will create:

```text
birdclef2026_resnet34_baseline.pt
submission.csv
```

## Useful Baseline Settings

The default configuration is intentionally small:

```python
epochs = 2
batch_size = 32
duration = 5.0
sample_rate = 32000
```

For a faster smoke test in Kaggle, change:

```python
cfg.max_train_rows = 1000
cfg.epochs = 1
```

For a stronger run, try:

```python
cfg.epochs = 5
cfg.max_train_rows = None
cfg.batch_size = 48
```

The best settings depend on Kaggle GPU memory and runtime limits.

## Important Limitation

This is a simple baseline, not a leaderboard-optimized solution. It should produce a valid `submission.csv`, but strong BirdCLEF solutions usually add heavier augmentations, pretrained audio/image encoders, better validation by site/date/source, class balancing, test-time augmentation, ensembling, and threshold calibration.
