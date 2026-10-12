"""
Milestone 4 - Knowledge distillation: CLIP teacher -> tiny CNN student.

The CLIP pipeline is accurate but slow on CPU (~7 patches/s). Here we
distill it into a small student CNN that mimics the teacher's patch
anomaly scores, so the system can one day run in real time on CPU/edge.

Phase 1 (teacher): extract patches from MVTec images, score each patch
                   with the CLIP pipeline, cache (patch, score) pairs.
Phase 2 (student): train a tiny CNN to mimic teacher scores (MSE loss).
Phase 3 (eval):    student-vs-teacher correlation on held-out patches,
                   FPS benchmark (student vs CLIP teacher), and a student
                   heatmap with pixel AUROC vs ground truth.

Usage:
    python src/distill.py --data-dir data --epochs 10
    (smoke test) python src/distill.py --data-dir data --epochs 2 --num-normal-imgs 4 --num-defect-imgs 4
"""

import argparse
import time
from pathlib import Path

import clip
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


class StudentCNN(nn.Module):
    """Tiny CNN (~200k params): 128x128 patch -> anomaly score in [0,1]."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),    # 64x64
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 32x32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


STUDENT_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PatchScoreDataset(Dataset):
    def __init__(self, patches, scores):
        self.patches = patches
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        return STUDENT_TF(self.patches[i]), self.scores[i]


def collect_images(category_dir: Path, num_normal: int, num_defect: int):
    normal = sorted((category_dir / "train" / "good").glob("*.png"))[:num_normal]
    defect = []
    for d in sorted((category_dir / "test").iterdir()):
        if d.is_dir() and d.name != "good":
            defect.extend(sorted(d.glob("*.png")))
    return normal, defect[:num_defect]


def extract_patches(image: Image.Image, patch_size=128, stride=128):
    w, h = image.size
    out = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            out.append(image.crop((x, y, x + patch_size, y + patch_size)))
    return out


@torch.no_grad()
def teacher_scores(patches, model, preprocess, normal_text, anomaly_text,
                   device, batch_size=32):
    scores = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(
            [preprocess(p) for p in patches[i:i + batch_size]]).to(device)
        feat = model.encode_image(batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        s_n = (feat @ normal_text.T).mean(dim=1)
        s_a = (feat @ anomaly_text.T).mean(dim=1)
        probs = torch.softmax(
            torch.stack([s_n, s_a], dim=1) * 100.0, dim=1)
        scores.extend(probs[:, 1].cpu().tolist())
    return np.array(scores, dtype=np.float32)


def build_heatmap(coords, scores, img_size, patch_size, sigma=8):
    w, h = img_size
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for (x, y), s in zip(coords, scores):
        acc[y:y + patch_size, x:x + patch_size] += s
        cnt[y:y + patch_size, x:x + patch_size] += 1
    return gaussian_filter(acc / np.maximum(cnt, 1), sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--num-normal-imgs", type=int, default=20)
    ap.add_argument("--num-defect-imgs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---------- Phase 1: teacher scores ----------
    category_dir = args.data_dir / args.category
    normal_imgs, defect_imgs = collect_images(
        category_dir, args.num_normal_imgs, args.num_defect_imgs)
    print(f"Images: {len(normal_imgs)} normal, {len(defect_imgs)} defective")

    model, preprocess = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        nt = model.encode_text(clip.tokenize(NORMAL_PROMPTS).to(device))
        nt = (nt / nt.norm(dim=-1, keepdim=True)).cpu()
        at = model.encode_text(clip.tokenize(ANOMALY_PROMPTS).to(device))
        at = (at / at.norm(dim=-1, keepdim=True)).cpu()

    all_patches, all_coords_img = [], []
    for p in normal_imgs + defect_imgs:
        img = Image.open(p).convert("RGB")
        all_patches.extend(extract_patches(img))
    print(f"Patches: {len(all_patches)}")

    t0 = time.time()
    t_scores = teacher_scores(all_patches, model, preprocess, nt, at,
                              device, args.batch_size)
    t_teacher = time.time() - t0
    print(f"Teacher scored {len(t_scores)} patches in {t_teacher:.1f}s "
          f"({len(t_scores) / t_teacher:.1f} patches/s)")

    # ---------- Phase 2: train student ----------
    idx = np.random.RandomState(0).permutation(len(all_patches))
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    train_ds = PatchScoreDataset([all_patches[i] for i in tr_idx],
                                 t_scores[tr_idx])
    val_ds = PatchScoreDataset([all_patches[i] for i in va_idx],
                               t_scores[va_idx])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    student = StudentCNN().to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        student.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(student(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        student.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                vt += loss_fn(student(x), y).item() * len(x)
                vn += len(x)
        print(f"Epoch {epoch}/{args.epochs}: train MSE={tot / n:.4f} "
              f"val MSE={vt / vn:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    torch.save(student.state_dict(), out_dir / "student.pt")
    print(f"Saved student weights: {out_dir / 'student.pt'}")

    # ---------- Phase 3: eval ----------
    student.eval()
    with torch.no_grad():
        preds, targets = [], []
        for x, y in val_dl:
            preds.extend(student(x.to(device)).cpu().tolist())
            targets.extend(y.tolist())
    preds, targets = np.array(preds), np.array(targets)
    corr = float(np.corrcoef(preds, targets)[0, 1])
    print(f"Student-vs-teacher correlation on held-out patches: {corr:.3f}")

    # FPS benchmark
    student.eval()
    dummy = torch.randn(args.batch_size, 3, 128, 128).to(device)
    with torch.no_grad():
        for _ in range(5):
            student(dummy)
        t0 = time.time()
        iters = 10
        for _ in range(iters):
            student(dummy)
    s_fps = iters * args.batch_size / (time.time() - t0)
    t_fps = len(t_scores) / t_teacher
    print(f"Student: {s_fps:.1f} patches/s vs teacher {t_fps:.1f} patches/s "
          f"({s_fps / t_fps:.1f}x faster)")

    # Student heatmap on a defective image + pixel AUROC
    test_dir = category_dir / "test"
    img_path, mask_path = None, None
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name != "good":
            f = sorted(d.glob("*.png"))
            if f:
                img_path = f[0]
                m = category_dir / "ground_truth" / d.name / f"{f[0].stem}_mask.png"
                mask_path = m if m.is_file() else None
                break
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    coords, patches = [], []
    for y in range(0, h - 128 + 1, 128):
        for x in range(0, w - 128 + 1, 128):
            coords.append((x, y))
            patches.append(image.crop((x, y, x + 128, y + 128)))
    with torch.no_grad():
        s_scores = []
        for i in range(0, len(patches), args.batch_size):
            xb = torch.stack([STUDENT_TF(p) for p in patches[i:i + args.batch_size]]).to(device)
            s_scores.extend(student(xb).cpu().tolist())
    heatmap = build_heatmap(coords, np.array(s_scores, dtype=np.float32),
                            (w, h), 128)
    auc = None
    if mask_path is not None:
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        hm = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            mask.shape[::-1], Image.BILINEAR)).astype(np.float32) / 255.0
        auc = roc_auc_score(mask.flatten(), hm.flatten())
        print(f"Student pixel AUROC vs ground truth: {auc:.3f}")

    fig, axes = plt.subplots(1, 3 if mask_path else 2, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    ttl = f"Student heatmap (max {max(s_scores):.2f})"
    axes[1].set_title(ttl)
    axes[1].axis("off")
    if mask_path is not None:
        axes[2].imshow(np.array(Image.open(mask_path).convert("L")) > 127,
                       cmap="gray")
        axes[2].set_title(f"Ground truth (AUROC {auc:.3f})")
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "student_heatmap.png", dpi=120)
    print(f"Saved: {out_dir / 'student_heatmap.png'}")
    print("DONE")


if __name__ == "__main__":
    main()
