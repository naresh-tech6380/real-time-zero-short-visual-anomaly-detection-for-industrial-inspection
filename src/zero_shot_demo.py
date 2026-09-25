"""
Milestone 1 - CLIP zero-shot anomaly scoring on MVTec AD (bottle).

Compares one normal vs one defective image using only text prompts
(no training, no labels) and prints which one CLIP thinks is anomalous.

Usage:
    python src/zero_shot_demo.py --data-dir data

Expected layout:
    data/
        bottle/
            test/
                good/
                broken_large/
                ...
"""

import argparse
from pathlib import Path

import clip
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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


def pick_images(category_dir: Path):
    """Return (one normal image, one defective image) from the test split."""
    test_dir = category_dir / "test"
    normal = sorted((test_dir / "good").glob("*.png"))[0]
    defective = None
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            files = sorted(defect_dir.glob("*.png"))
            if files:
                defective = files[0]
                break
    if defective is None:
        raise FileNotFoundError(f"No defective images found under {test_dir}")
    return normal, defective


@torch.no_grad()
def anomaly_score(image, model, preprocess, normal_text, anomaly_text, device):
    """Probability that the image is anomalous, from CLIP text similarity."""
    x = preprocess(image).unsqueeze(0).to(device)
    img_feat = model.encode_image(x)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    sim_normal = (img_feat @ normal_text.T).mean().item()
    sim_anomaly = (img_feat @ anomaly_text.T).mean().item()
    logits = np.array([sim_normal, sim_anomaly]) * 100.0  # CLIP logit scale
    exp = np.exp(logits - logits.max())
    return float(exp[1] / exp.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Folder containing the category folder (default: data)")
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()

    category_dir = args.data_dir / args.category
    if not category_dir.is_dir():
        raise SystemExit(
            f"Dataset not found at {category_dir}.\n"
            f"Download the MVTec AD '{args.category}' category and extract it "
            f"under {args.data_dir}/ (see README)."
        )

    normal_path, defective_path = pick_images(category_dir)
    print(f"Normal image:    {normal_path}")
    print(f"Defective image: {defective_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (first run downloads the CLIP model, ~350 MB)")
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        normal_text = model.encode_text(clip.tokenize(NORMAL_PROMPTS).to(device))
        normal_text = normal_text / normal_text.norm(dim=-1, keepdim=True)
        anomaly_text = model.encode_text(clip.tokenize(ANOMALY_PROMPTS).to(device))
        anomaly_text = anomaly_text / anomaly_text.norm(dim=-1, keepdim=True)

    normal_img = Image.open(normal_path).convert("RGB")
    defective_img = Image.open(defective_path).convert("RGB")

    s_normal = anomaly_score(normal_img, model, preprocess,
                             normal_text, anomaly_text, device)
    s_defect = anomaly_score(defective_img, model, preprocess,
                             normal_text, anomaly_text, device)

    print(f"\nAnomaly score (normal image):    {s_normal:.3f}")
    print(f"Anomaly score (defective image): {s_defect:.3f}")
    if s_defect > s_normal:
        print("PASS: defective image scored higher than normal.")
    else:
        print("CHECK: normal scored higher - inspect the images/prompts.")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(normal_img)
    axes[0].set_title(f"Normal (score {s_normal:.3f})")
    axes[0].axis("off")
    axes[1].imshow(defective_img)
    axes[1].set_title(f"Defective (score {s_defect:.3f})")
    axes[1].axis("off")
    out_path = out_dir / "zero_shot_demo.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
