"""
Milestone 3 - Streaming patch memory + live anomaly demo.

Processes a stream of frames (webcam, video file, or image folder) and
maintains a rolling, position-aware memory bank of patch embeddings from
recent normal frames (PatchCore-style): each patch is compared only
against normal patches previously seen at the SAME grid cell, so a defect
stands out against the normal appearance of its own location.

The per-patch memory distance is the primary anomaly score; the CLIP
zero-shot text similarity (normal vs anomaly prompts) is reported as a
reference signal. The verdict threshold is auto-calibrated from the
seeded normal frames (leave-one-out: worst normal-vs-normal distance
plus a margin) -- anything stranger than the strangest normal example
is flagged DEFECT. Frames judged normal refresh the memory, so the
system adapts to gradual appearance changes on the line.

Usage:
    python src/streaming_demo.py --source 0 --init-dir data/bottle/train/good
    python src/streaming_demo.py --source data/bottle/test/good --no-display --max-frames 10
"""

import argparse
import time
from collections import deque
from pathlib import Path

import clip
import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

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


def iter_frames(source: Path):
    """Yield (name, PIL image) from an image folder, video file, or webcam."""
    if source.is_dir():
        files = sorted(source.glob("*.png")) + sorted(source.glob("*.jpg"))
        if not files:
            raise SystemExit(f"No images found in {source}")
        for p in files:
            yield p.name, Image.open(p).convert("RGB")
    else:
        try:
            cam_index = int(str(source))
        except ValueError:
            cam_index = None
        cap = cv2.VideoCapture(cam_index if cam_index is not None else str(source))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video source: {source}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"frame_{i:04d}", Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()


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
def encode_patches(patches, model, preprocess, device, batch_size=32):
    feats = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(
            [preprocess(p) for p in patches[i:i + batch_size]]).to(device)
        f = model.encode_image(batch)
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def score_frame(patch_feats, normal_text, anomaly_text, memory):
    """Per-patch memory distance (primary signal) and CLIP text score.

    Returns (mem_score, text_score). mem_score is the position-aware
    PatchCore-style distance: 1 minus the best cosine similarity to normal
    patches previously seen at the SAME grid cell. It is 0.0 where the
    memory has no history yet. text_score is the CLIP zero-shot
    normal-vs-anomaly probability, kept as a reference signal.
    """
    sim_normal = (patch_feats @ normal_text.T).mean(dim=1)
    sim_anomaly = (patch_feats @ anomaly_text.T).mean(dim=1)
    text_score = torch.softmax(
        torch.stack([sim_normal, sim_anomaly], dim=1) * 100.0, dim=1)[:, 1]
    mem_score = torch.zeros_like(text_score)
    for i, pf in enumerate(patch_feats):
        if i < len(memory) and len(memory[i]) > 0:
            mem = torch.stack(list(memory[i]))  # (K, D), already normalized
            mem_score[i] = 1.0 - (pf @ mem.T).max()
    return mem_score, text_score


def calibrate_threshold(memory, margin=0.03):
    """Leave-one-out threshold from the seeded normal patches.

    Each seed patch is scored against the OTHER seed patches at its grid
    cell; the threshold is the worst normal-vs-normal distance plus a
    margin. Anything stranger than the strangest normal example seen
    during seeding is flagged as a defect. Returns None when there are
    too few seed patches to calibrate.
    """
    scores = []
    for cell in memory:
        if len(cell) < 2:
            continue
        m = torch.stack(list(cell))
        sim = m @ m.T
        sim.fill_diagonal_(-1.0)
        scores.extend((1.0 - sim.max(dim=1).values).tolist())
    if not scores:
        return None
    return max(scores) + margin


def build_heatmap(coords, scores, img_size, patch_size, sigma=8):
    w, h = img_size
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for (x, y), s in zip(coords, scores):
        acc[y:y + patch_size, x:x + patch_size] += s
        cnt[y:y + patch_size, x:x + patch_size] += 1
    return gaussian_filter(acc / np.maximum(cnt, 1), sigma=sigma)


def overlay_image(image, heatmap):
    """PIL image with jet heatmap blended on top (for display / saving)."""
    hm = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(
        cv2.resize(hm, image.size), cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, 0.6, colored, 0.4, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="Webcam index (0), video file, or image folder")
    ap.add_argument("--init-dir", type=Path, default=None,
                    help="Folder of known-normal images to seed the memory")
    ap.add_argument("--init-frames", type=int, default=3)
    ap.add_argument("--patch-size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=112,
                    help="Overlapping stride (stride < patch-size) gives "
                         "denser coverage so small defects are not split "
                         "across patch boundaries.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--memory-size", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = unlimited)")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path("outputs/stream"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model_device = next(model.parameters()).device
    with torch.no_grad():
        normal_text = model.encode_text(
            clip.tokenize(NORMAL_PROMPTS).to(device))
        normal_text = (normal_text / normal_text.norm(dim=-1, keepdim=True)).cpu()
        anomaly_text = model.encode_text(
            clip.tokenize(ANOMALY_PROMPTS).to(device))
        anomaly_text = (anomaly_text / anomaly_text.norm(dim=-1, keepdim=True)).cpu()

    memory = []  # position-aware: one deque of normal embeddings per grid cell

    def memory_add(feats):
        """Add one frame's patch embeddings to their matching grid cells."""
        if len(memory) != len(feats):
            per_pos = max(1, args.memory_size // max(1, len(feats)))
            memory.clear()
            memory.extend(deque(maxlen=per_pos) for _ in range(len(feats)))
        for cell, f in zip(memory, feats):
            cell.append(f)

    def memory_total():
        return sum(len(cell) for cell in memory)

    use_memory = args.init_dir is not None and args.init_dir.is_dir()
    if use_memory:
        seed_files = (sorted(args.init_dir.glob("*.png")) +
                      sorted(args.init_dir.glob("*.jpg")))[:args.init_frames]
        for p in seed_files:
            img = Image.open(p).convert("RGB")
            patches, _, _, _ = extract_patches(img, args.patch_size, args.stride)
            feats = encode_patches(patches, model, preprocess,
                                   model_device, args.batch_size)
            memory_add(feats)
        print(f"Memory seeded with {memory_total()} normal patches "
              f"from {len(seed_files)} images ({len(memory)} grid cells).")
        threshold = calibrate_threshold(memory)
        if threshold is None:
            threshold = args.threshold
            print(f"Too few seed patches to calibrate; "
                  f"falling back to --threshold={threshold:.3f}.")
        else:
            print(f"Calibrated threshold: {threshold:.3f} "
                  f"(leave-one-out over seed patches).")
    else:
        threshold = args.threshold
        print(f"No --init-dir given: text-only mode, "
              f"threshold={threshold:.3f}.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    n_defects = 0
    t_start = time.time()

    for name, image in iter_frames(args.source):
        patches, coords, img_size, ps = extract_patches(
            image, args.patch_size, args.stride)
        feats = encode_patches(patches, model, preprocess,
                               model_device, args.batch_size)
        mem_score, text_score = score_frame(feats, normal_text, anomaly_text,
                                            memory)
        # Memory distance drives the verdict once normal history exists;
        # otherwise fall back to the CLIP text score.
        patch_score = mem_score if memory_total() > 0 else text_score
        frame_score = float(patch_score.max())
        is_defect = frame_score > threshold
        n_frames += 1
        n_defects += int(is_defect)
        verdict = "DEFECT" if is_defect else "OK"
        print(f"[{n_frames:03d}] {name}: score={frame_score:.3f} "
              f"(text={float(text_score.max()):.3f}) "
              f"-> {verdict} (memory={memory_total()})")

        if not is_defect:
            memory_add(feats)  # adapt to the line

        heatmap = build_heatmap(coords, patch_score.numpy(), img_size, ps)
        vis = overlay_image(image, heatmap)
        cv2.imwrite(str(args.save_dir / f"{Path(name).stem}_{verdict}.jpg"), vis)

        if not args.no_display:
            label = f"{verdict} {frame_score:.2f}"
            color = (0, 0, 255) if is_defect else (0, 255, 0)
            cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color, 3)
            cv2.imshow("CLIP-StreamAD live", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames and n_frames >= args.max_frames:
            break

    if not args.no_display:
        cv2.destroyAllWindows()
    dt = time.time() - t_start
    print(f"\nDone: {n_frames} frames, {n_defects} flagged DEFECT, "
          f"{n_frames / dt:.2f} FPS on {device}.")


if __name__ == "__main__":
    main()
