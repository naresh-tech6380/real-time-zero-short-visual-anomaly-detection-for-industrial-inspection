"""
Milestone 2 - Patch-level anomaly localization with CLIP (zero-shot).

Slides a window over the image, scores every patch with CLIP using only
text prompts (no training, no labels), and builds an anomaly heatmap that
shows WHERE the defect is. Reports pixel-level AUROC against the MVTec
ground-truth mask.

Usage:
    python src/patch_localize.py --data-dir data --category bottle
"""

import argparse
import time
from pathlib import Path

import clip
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

NORMAL_PROMPTS = [
    "a photo of a normal bottle",
    "a photo of a flawless bottle",
    "a photo of a bottle without any defects",
]

ANOMALY_PROMPTS = [
    "a photo of a damaged bottle",
    "a photo of a defective bottle",
    "a photo of a bottle with scratches or cracks",
]


def find_defective(category_dir: Path):
    """Return (defective image path, ground-truth mask path or None)."""
    test_dir = category_dir / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            files = sorted(defect_dir.glob("*.png"))
            if files:
                img = files[0]
                mask = (category_dir / "ground_truth" / defect_dir.name
                        / f"{img.stem}_mask.png")
                return img, mask if mask.is_file() else None
    raise FileNotFoundError(f"No defective images found under {test_dir}")


def extract_patches(image: Image.Image, patch_size: int, stride: int):
    w, h = image.size
    patch_size = min(patch_size, w, h)
    patches, coords = [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image.crop((x, y, x + patch_size, y + patch_size)))
            coords.append((x, y))
    return patches, coords, (w, h), patch_size


@torch.no_grad()
def score_patches(patches, model, preprocess, normal_text, anomaly_text,
                  device, batch_size=32):
    """Anomaly probability per patch, scored in batches for speed."""
    scores = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(
            [preprocess(p) for p in patches[i:i + batch_size]]).to(device)
        feat = model.encode_image(batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sim_normal = (feat @ normal_text.T).mean(dim=1)
        sim_anomaly = (feat @ anomaly_text.T).mean(dim=1)
        probs = torch.softmax(
            torch.stack([sim_normal, sim_anomaly], dim=1) * 100.0, dim=1)
        scores.extend(probs[:, 1].cpu().tolist())
    return np.array(scores, dtype=np.float32)


def build_heatmap(coords, scores, img_size, patch_size, sigma=8):
    """Average overlapping patch scores into a smooth full-image heatmap."""
    w, h = img_size
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for (x, y), s in zip(coords, scores):
        acc[y:y + patch_size, x:x + patch_size] += s
        cnt[y:y + patch_size, x:x + patch_size] += 1
    heatmap = acc / np.maximum(cnt, 1)
    return gaussian_filter(heatmap, sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Folder containing the category folder (default: data)")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    category_dir = args.data_dir / args.category
    img_path, mask_path = find_defective(category_dir)
    print(f"Image: {img_path}")
    print(f"Mask:  {mask_path}")

    image = Image.open(img_path).convert("RGB")
    patches, coords, img_size, ps = extract_patches(
        image, args.patch_size, args.stride)
    print(f"Patches: {len(patches)} ({ps}x{ps}, stride {args.stride})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        normal_text = model.encode_text(clip.tokenize(NORMAL_PROMPTS).to(device))
        normal_text = normal_text / normal_text.norm(dim=-1, keepdim=True)
        anomaly_text = model.encode_text(clip.tokenize(ANOMALY_PROMPTS).to(device))
        anomaly_text = anomaly_text / anomaly_text.norm(dim=-1, keepdim=True)

    t0 = time.time()
    scores = score_patches(patches, model, preprocess, normal_text,
                           anomaly_text, device, args.batch_size)
    dt = time.time() - t0
    print(f"Scored {len(scores)} patches in {dt:.1f}s "
          f"({len(scores) / dt:.1f} patches/s)")

    heatmap = build_heatmap(coords, scores, img_size, ps)
    img_score = float(scores.max())
    print(f"Image anomaly score (max patch): {img_score:.3f}")

    mask = None
    auc = None
    if mask_path is not None:
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        hm = np.array(Image.fromarray(
            (heatmap * 255).astype(np.uint8)).resize(
            mask.shape[::-1], Image.BILINEAR)).astype(np.float32) / 255.0
        auc = roc_auc_score(mask.flatten(), hm.flatten())
        print(f"Pixel AUROC vs ground truth: {auc:.3f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ncols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Anomaly heatmap (max {img_score:.2f})")
    axes[1].axis("off")
    if mask is not None:
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title(f"Ground truth (AUROC {auc:.3f})")
        axes[2].axis("off")
    out_path = out_dir / "patch_localize.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
