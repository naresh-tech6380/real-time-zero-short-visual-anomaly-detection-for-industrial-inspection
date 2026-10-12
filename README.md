# CLIP-StreamAD (starter)

Real-Time Zero-Shot Visual Anomaly Detection for Industrial Inspection.
Milestone 1: CLIP zero-shot anomaly scoring on MVTec AD — no training, no labels.

## Setup (Windows)

1. Install Python 3.10+ from python.org (tick "Add python.exe to PATH").
2. Install Git from git-scm.com (needed to install the CLIP package).
3. Clone this repo and enter it:
   ```
   git clone https://github.com/naresh-tech6380/real-time-zero-short-visual-anomaly-detection-for-industrial-inspection.git
   cd real-time-zero-short-visual-anomaly-detection-for-industrial-inspection
   ```
4. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install dependencies (CPU-only PyTorch is all we need — no GPU):
   ```
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   pip install -r requirements.txt
   ```
   (Install torch from the CPU index first: the default PyPI `torch`
   wheel bundles CUDA and is ~2.5 GB. The `pip install -r` step then
   skips torch/torchvision since they are already satisfied.)
   First run downloads the CLIP model (~350 MB) — needs internet once.

## Dataset

1. Download MVTec AD (free, CC BY-NC-SA 4.0) from the official page:
   https://www.mvtec.com/company/research/datasets/mvtec-ad
   (fill the download form on that page; the old direct per-category
   links are no longer live). You only need the `bottle` category.
2. Extract it into `data/` so the layout is:
   ```
   data/
       bottle/
           test/
               good/
               broken_large/
               ...
   ```

## Run milestone 1

```
python src/zero_shot_demo.py --data-dir data
```

It prints anomaly scores for one normal vs one defective bottle image and
saves `outputs/zero_shot_demo.png`. Milestone passes if the defective image
scores higher than the normal one.

## Run milestone 2

```
python src/patch_localize.py --data-dir data
```

It slides a 128x128 window (stride 64) over a defective bottle image, scores
every patch with CLIP, and builds an anomaly heatmap showing where the defect
is. It prints the pixel-level AUROC against the MVTec ground-truth mask and
saves `outputs/patch_localize.png` (original | heatmap overlay | ground truth).

## Run milestone 3

```
python src/streaming_demo.py --source 0 --init-dir data/bottle/train/good
```

Simulates a production line: each frame is split into overlapping patches
(224x224, stride 112) and every patch is compared against a rolling,
position-aware memory of normal patches seen at the same grid cell
(PatchCore-style). The per-patch memory distance is the anomaly score; the
CLIP zero-shot text score is printed as a reference. The verdict threshold
is auto-calibrated from the seeded normal frames (leave-one-out: worst
normal-vs-normal distance + margin), so there is nothing to hand-tune.
Frames judged OK refresh the memory (adapts to gradual changes);
frames judged DEFECT do not (no pollution). Each frame's verdict and
heatmap are saved to `outputs/stream/`.

- `--source`: webcam index (e.g. `0`), a video file, or a folder of images.
- `--init-dir`: folder of known-normal images used to seed the memory and
  calibrate the threshold. Without it, the script falls back to CLIP
  text-only scoring with `--threshold` (default 0.7).
- `--no-display --max-frames N`: headless mode, stop after N frames.

Note: on CPU this runs at a few frames per minute (CLIP encoding is the
bottleneck) -- it demonstrates the streaming architecture, not real-time
speed. Real-time performance comes later via the distilled student model.
