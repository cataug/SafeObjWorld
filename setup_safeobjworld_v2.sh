#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$HOME/SafeObjWorld"
CODE_DIR="$PROJECT_ROOT/code"
DATA_ROOT="$PROJECT_ROOT/data/DAVIS2017"
RUN_ROOT="$PROJECT_ROOT/runs_v2_full"
VENV_DIR="$PROJECT_ROOT/.venv_safeobjworld"

mkdir -p "$CODE_DIR" "$RUN_ROOT"

echo "============================================================"
echo "SafeObjWorld V2 setup"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "============================================================"

if [ ! -d "$DATA_ROOT/DAVIS/JPEGImages/480p" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/JPEGImages/480p"
  exit 1
fi

if [ ! -d "$DATA_ROOT/DAVIS/Annotations/480p" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/Annotations/480p"
  exit 1
fi

if [ ! -f "$DATA_ROOT/DAVIS/ImageSets/2017/train.txt" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/ImageSets/2017/train.txt"
  exit 1
fi

if [ ! -f "$DATA_ROOT/DAVIS/ImageSets/2017/val.txt" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/ImageSets/2017/val.txt"
  exit 1
fi

echo "[OK] DAVIS2017 structure found."

# ============================================================
# Environment
# ============================================================

if [ "${SKIP_ENV:-0}" != "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "[env] creating $VENV_DIR"
    if command -v python3.10 >/dev/null 2>&1; then
      python3.10 -m venv "$VENV_DIR"
    elif command -v python3.11 >/dev/null 2>&1; then
      python3.11 -m venv "$VENV_DIR"
    else
      python3 -m venv "$VENV_DIR"
    fi
  fi

  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel

  echo "[env] installing torch cu121"
  pip install --upgrade --force-reinstall \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

  echo "[env] installing deps"
  pip install --upgrade numpy pandas pillow opencv-python tqdm scikit-learn matplotlib
else
  source "$VENV_DIR/bin/activate"
fi

echo "============================================================"
echo "Python/CUDA check"
echo "============================================================"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

# ============================================================
# Runner
# ============================================================

cat > "$CODE_DIR/safeobjworld_v2.py" <<'PY'
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

try:
    import cv2
except Exception:
    cv2 = None

import torch
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png"}


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def locate_davis_root(dataset_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    if (dataset_root / "DAVIS").exists():
        return dataset_root / "DAVIS"
    if (dataset_root / "JPEGImages").exists() and (dataset_root / "Annotations").exists():
        return dataset_root
    raise FileNotFoundError(f"Cannot locate DAVIS root under {dataset_root}")


def read_split(davis_root, year="2017", split="train"):
    p = Path(davis_root) / "ImageSets" / str(year) / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def sorted_files(p, exts):
    p = Path(p)
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_label_mask(path):
    return np.array(Image.open(path), dtype=np.int32)


def resize_rgb_pil(img, size):
    return img.resize((size, size), Image.BILINEAR)


def resize_mask_np(mask, size):
    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    pil = pil.resize((size, size), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


def bbox_from_mask(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_geom(mask):
    mask = mask.astype(bool)
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(8, dtype=np.float32)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = xs.mean() / max(w - 1, 1)
    cy = ys.mean() / max(h - 1, 1)
    area = mask.mean()
    bw = (x2 - x1 + 1) / max(w, 1)
    bh = (y2 - y1 + 1) / max(h, 1)

    return np.array([
        cx,
        cy,
        x1 / max(w - 1, 1),
        y1 / max(h - 1, 1),
        x2 / max(w - 1, 1),
        y2 / max(h - 1, 1),
        area,
        bw * bh,
    ], dtype=np.float32)


def apply_occlusion(images, masks, rng, severity=0.45):
    images = [im.copy() for im in images]
    masks = [m.copy() for m in masks]

    ref = masks[-1]
    bbox = bbox_from_mask(ref)
    if bbox is None:
        return images, masks

    x1, y1, x2, y2 = bbox
    h, w = ref.shape
    bw = max(2, x2 - x1 + 1)
    bh = max(2, y2 - y1 + 1)

    occ_w = max(2, int(bw * severity))
    occ_h = max(2, int(bh * severity))

    ox1 = int(np.clip(x1 + rng.uniform(0.05, 0.55) * bw, 0, w - 1))
    oy1 = int(np.clip(y1 + rng.uniform(0.05, 0.55) * bh, 0, h - 1))
    ox2 = int(np.clip(ox1 + occ_w, 0, w))
    oy2 = int(np.clip(oy1 + occ_h, 0, h))

    for i in range(len(images)):
        arr = np.array(images[i]).copy()
        arr[oy1:oy2, ox1:ox2, :] = 0
        images[i] = Image.fromarray(arr.astype(np.uint8))
        masks[i][oy1:oy2, ox1:ox2] = 0.0

    return images, masks


def apply_distractor(images, masks, rng):
    images = [im.copy() for im in images]
    masks = [m.copy() for m in masks]

    ref = masks[-1]
    bbox = bbox_from_mask(ref)
    if bbox is None:
        return images, masks

    x1, y1, x2, y2 = bbox
    h, w = ref.shape
    bw = max(4, x2 - x1 + 1)
    bh = max(4, y2 - y1 + 1)

    dx = int(rng.choice([-1, 1]) * rng.uniform(0.6, 1.3) * bw)
    dy = int(rng.choice([-1, 1]) * rng.uniform(0.1, 0.8) * bh)

    tx1 = int(np.clip(x1 + dx, 0, w - 1))
    ty1 = int(np.clip(y1 + dy, 0, h - 1))
    tx2 = int(np.clip(tx1 + bw, 0, w))
    ty2 = int(np.clip(ty1 + bh, 0, h))

    if tx2 <= tx1 or ty2 <= ty1:
        return images, masks

    for i in range(len(images)):
        arr = np.array(images[i]).copy()
        color = arr[ref.astype(bool)].mean(axis=0) if ref.sum() > 0 else np.array([120, 120, 120])
        arr[ty1:ty2, tx1:tx2, :] = color.astype(np.uint8)
        images[i] = Image.fromarray(arr.astype(np.uint8))

    return images, masks


def apply_stress(images, masks, stress, rng):
    if stress in [None, "clean"]:
        return images, masks

    images = [im.copy() for im in images]
    masks = [m.copy() for m in masks]

    if stress == "blur":
        return [im.filter(ImageFilter.GaussianBlur(radius=2.0)) for im in images], masks

    if stress == "lowlight":
        return [ImageEnhance.Brightness(im).enhance(0.45) for im in images], masks

    if stress == "noise":
        out = []
        for im in images:
            arr = np.array(im).astype(np.float32)
            arr = np.clip(arr + rng.normal(0, 18, arr.shape), 0, 255)
            out.append(Image.fromarray(arr.astype(np.uint8)))
        return out, masks

    if stress == "occlusion":
        return apply_occlusion(images, masks, rng, severity=0.45)

    if stress == "frame_drop":
        if len(images) >= 3:
            mid = len(images) // 2
            images[mid] = images[0].copy()
            masks[mid] = masks[0].copy()
        return images, masks

    if stress == "distractor":
        return apply_distractor(images, masks, rng)

    raise ValueError(f"Unknown stress: {stress}")


# ============================================================
# Dataset
# ============================================================

class DAVISObjectFutureDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split="train",
        year="2017",
        prefix_len=3,
        max_h=5,
        image_size=192,
        stress="clean",
        random_stress=False,
        max_samples=0,
        max_objects_per_frame=20,
        seed=42,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.davis_root = locate_davis_root(self.dataset_root)
        self.split = split
        self.year = str(year)
        self.prefix_len = int(prefix_len)
        self.max_h = int(max_h)
        self.image_size = int(image_size)
        self.stress = stress
        self.random_stress = bool(random_stress)
        self.max_samples = int(max_samples)
        self.max_objects_per_frame = int(max_objects_per_frame)
        self.seed = int(seed)

        self.img_root = self.davis_root / "JPEGImages" / "480p"
        self.ann_root = self.davis_root / "Annotations" / "480p"

        seqs = read_split(self.davis_root, self.year, self.split)
        self.samples = []

        for seq in seqs:
            img_dir = self.img_root / seq
            ann_dir = self.ann_root / seq
            if not img_dir.exists() or not ann_dir.exists():
                continue

            imgs = sorted_files(img_dir, IMAGE_EXTS)
            anns = sorted_files(ann_dir, MASK_EXTS)

            stem_to_img = {p.stem: p for p in imgs}
            stem_to_ann = {p.stem: p for p in anns}
            stems = sorted(set(stem_to_img) & set(stem_to_ann))

            if len(stems) < self.prefix_len + self.max_h:
                continue

            for start in range(0, len(stems) - self.prefix_len - self.max_h + 1):
                last_stem = stems[start + self.prefix_len - 1]
                label = load_label_mask(stem_to_ann[last_stem])
                obj_ids = [int(x) for x in np.unique(label) if int(x) > 0]
                obj_ids = obj_ids[: self.max_objects_per_frame]

                for oid in obj_ids:
                    self.samples.append({
                        "seq": seq,
                        "start": start,
                        "oid": oid,
                        "stems": stems,
                        "img_paths": [stem_to_img[s] for s in stems],
                        "ann_paths": [stem_to_ann[s] for s in stems],
                    })

        if self.max_samples and self.max_samples > 0 and len(self.samples) > self.max_samples:
            rng = random.Random(self.seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

        if len(self.samples) == 0:
            raise RuntimeError("No samples created. Check dataset root/split/horizon.")

    def __len__(self):
        return len(self.samples)

    def _load_object_mask(self, ann_path, oid):
        label = load_label_mask(ann_path)
        return (label == int(oid)).astype(np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        start = int(s["start"])
        oid = int(s["oid"])

        prefix_images = []
        prefix_masks = []

        for j in range(start, start + self.prefix_len):
            prefix_images.append(load_rgb(s["img_paths"][j]))
            prefix_masks.append(self._load_object_mask(s["ann_paths"][j], oid))

        target_masks = []
        for h in range(1, self.max_h + 1):
            j = start + self.prefix_len - 1 + h
            target_masks.append(self._load_object_mask(s["ann_paths"][j], oid))

        rng = np.random.default_rng(self.seed + idx)

        if self.random_stress:
            stress = str(rng.choice(["clean", "occlusion", "blur", "frame_drop", "lowlight"]))
        else:
            stress = self.stress

        prefix_images, prefix_masks = apply_stress(prefix_images, prefix_masks, stress, rng)

        x_seq = []
        geom_seq = []

        for im, m in zip(prefix_images, prefix_masks):
            im = resize_rgb_pil(im, self.image_size)
            im_np = np.array(im).astype(np.float32) / 255.0
            im_np = np.transpose(im_np, (2, 0, 1))

            m_rs = resize_mask_np(m, self.image_size)
            geom_seq.append(mask_geom(m_rs))

            x = np.concatenate([im_np, m_rs[None, :, :]], axis=0)
            x_seq.append(x)

        y = []
        for m in target_masks:
            m_rs = resize_mask_np(m, self.image_size)
            y.append(m_rs[None, :, :])

        x_seq = torch.tensor(np.stack(x_seq, axis=0), dtype=torch.float32)
        geom_seq = torch.tensor(np.stack(geom_seq, axis=0), dtype=torch.float32)
        y = torch.tensor(np.stack(y, axis=0), dtype=torch.float32)

        return {
            "x_seq": x_seq,
            "geom": geom_seq,
            "target": y,
            "seq": s["seq"],
            "start": int(start),
            "oid": int(oid),
            "stress": stress,
            "sample_id": f"{s['seq']}:{start}:{oid}",
        }


# ============================================================
# Models
# ============================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=pad)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMWorld(nn.Module):
    def __init__(
        self,
        in_ch=4,
        hidden=64,
        max_h=5,
        image_size=192,
        dropout=0.0,
        input_mode="rgb_mask",
    ):
        super().__init__()
        self.max_h = int(max_h)
        self.image_size = int(image_size)
        self.dropout_p = float(dropout)
        self.input_mode = input_mode

        if input_mode == "rgb_mask":
            real_in = 4
        elif input_mode == "rgb_only":
            real_in = 3
        elif input_mode == "mask_only":
            real_in = 1
        else:
            raise ValueError(input_mode)

        self.enc = nn.Sequential(
            nn.Conv2d(real_in, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.cell = ConvLSTMCell(hidden, hidden)
        self.roll_cell = ConvLSTMCell(hidden, hidden)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden, 48, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 3, padding=1),
        )

    def select_input(self, x):
        if self.input_mode == "rgb_mask":
            return x
        if self.input_mode == "rgb_only":
            return x[:, :3]
        if self.input_mode == "mask_only":
            return x[:, 3:4]
        raise ValueError(self.input_mode)

    def forward(self, x_seq, geom=None, sample=False):
        b, l, c, h, w = x_seq.shape
        state = None
        last_feat = None

        # Enable dropout during MC evaluation.
        if sample and self.dropout_p > 0:
            was_training = self.training
            self.train()
        else:
            was_training = None

        for t in range(l):
            xt = self.select_input(x_seq[:, t])
            feat = self.enc(xt)
            last_feat = feat
            if state is None:
                zeros = torch.zeros(
                    b, feat.shape[1], feat.shape[2], feat.shape[3],
                    device=feat.device,
                    dtype=feat.dtype,
                )
                state = (zeros, zeros)
            state = self.cell(feat, state)

        outs = []
        inp = last_feat

        for _ in range(self.max_h):
            state = self.roll_cell(inp, state)
            logits = self.dec(state[0])
            if logits.shape[-1] != self.image_size or logits.shape[-2] != self.image_size:
                logits = F.interpolate(
                    logits,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            outs.append(logits)
            inp = torch.zeros_like(inp)

        if was_training is not None:
            self.train(was_training)

        return torch.stack(outs, dim=1)


class ObjectTokenWorld(nn.Module):
    def __init__(
        self,
        image_size=192,
        in_ch=4,
        geom_dim=8,
        use_geom=True,
        latent=256,
        max_h=5,
        stochastic=True,
        dropout=0.15,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.max_h = int(max_h)
        self.use_geom = bool(use_geom)
        self.stochastic = bool(stochastic)
        self.dropout_p = float(dropout)
        self.latent = int(latent)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        token_dim = 128 + (geom_dim if self.use_geom else 0)
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, latent),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(input_size=latent, hidden_size=latent, batch_first=True)
        self.roll = nn.GRUCell(latent, latent)
        self.step_token = nn.Parameter(torch.randn(1, latent) * 0.02)
        self.drop = nn.Dropout(dropout)

        self.spatial = max(6, self.image_size // 16)
        self.fc = nn.Sequential(
            nn.Linear(latent, 128 * self.spatial * self.spatial),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def encode_tokens(self, x_seq, geom):
        b, l, c, h, w = x_seq.shape
        tokens = []
        for t in range(l):
            feat = self.cnn(x_seq[:, t]).flatten(1)
            if self.use_geom:
                tok = torch.cat([feat, geom[:, t]], dim=1)
            else:
                tok = feat
            tokens.append(self.token_proj(tok))
        return torch.stack(tokens, dim=1)

    def decode_mask(self, z):
        b = z.shape[0]
        y = self.fc(z).view(b, 128, self.spatial, self.spatial)
        logits = self.dec(y)
        if logits.shape[-1] != self.image_size or logits.shape[-2] != self.image_size:
            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def forward(self, x_seq, geom, sample=False):
        if sample and self.dropout_p > 0:
            was_training = self.training
            self.train()
        else:
            was_training = None

        tokens = self.encode_tokens(x_seq, geom)
        _, h = self.gru(tokens)
        z = h[-1]

        outs = []
        inp = self.step_token.repeat(z.shape[0], 1)

        for _ in range(self.max_h):
            z = self.roll(inp, z)
            if self.stochastic and (self.training or sample):
                z_dec = self.drop(z)
                z_dec = z_dec + 0.025 * torch.randn_like(z_dec)
            else:
                z_dec = z
            outs.append(self.decode_mask(z_dec))
            inp = self.step_token.repeat(z.shape[0], 1)

        if was_training is not None:
            self.train(was_training)

        return torch.stack(outs, dim=1)


def build_model(method, image_size, max_h):
    if method in [
        "convlstm_base",
        "convlstm_stress",
        "safeobjworld_r_full",
        "safeobjworld_r_no_temporal",
        "safeobjworld_r_no_split",
        "safeobjworld_r_no_area",
        "safeobjworld_r_no_stress",
        "safeobjworld_r_det",
        "safeobjworld_r_image_only",
        "safeobjworld_r_mask_only",
    ]:
        dropout = 0.0
        input_mode = "rgb_mask"

        if method.startswith("safeobjworld_r"):
            dropout = 0.15

        if method == "safeobjworld_r_det":
            dropout = 0.0

        if method == "safeobjworld_r_image_only":
            input_mode = "rgb_only"

        if method == "safeobjworld_r_mask_only":
            input_mode = "mask_only"

        return ConvLSTMWorld(
            in_ch=4,
            hidden=64,
            max_h=max_h,
            image_size=image_size,
            dropout=dropout,
            input_mode=input_mode,
        )

    if method == "object_token_full":
        return ObjectTokenWorld(
            image_size=image_size,
            max_h=max_h,
            use_geom=True,
            stochastic=True,
            dropout=0.15,
        )

    if method == "object_token_det":
        return ObjectTokenWorld(
            image_size=image_size,
            max_h=max_h,
            use_geom=True,
            stochastic=False,
            dropout=0.0,
        )

    if method == "object_token_no_geom":
        return ObjectTokenWorld(
            image_size=image_size,
            max_h=max_h,
            use_geom=False,
            stochastic=True,
            dropout=0.15,
        )

    raise ValueError(f"Unknown method: {method}")


def method_config(method):
    cfg = {
        "stress_train": False,
        "temporal_weight": 0.0,
        "split_weight": 0.0,
        "area_weight": 0.0,
        "mc_eval": False,
        "eval_k": 1,
    }

    if method == "convlstm_base":
        return cfg

    if method == "convlstm_stress":
        cfg["stress_train"] = True
        return cfg

    if method == "safeobjworld_r_full":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.005,
            "area_weight": 0.10,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method == "safeobjworld_r_no_temporal":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.0,
            "split_weight": 0.005,
            "area_weight": 0.10,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method == "safeobjworld_r_no_split":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.0,
            "area_weight": 0.10,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method == "safeobjworld_r_no_area":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.005,
            "area_weight": 0.0,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method == "safeobjworld_r_no_stress":
        cfg.update({
            "stress_train": False,
            "temporal_weight": 0.05,
            "split_weight": 0.005,
            "area_weight": 0.10,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method == "safeobjworld_r_det":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.005,
            "area_weight": 0.10,
            "mc_eval": False,
            "eval_k": 1,
        })
        return cfg

    if method == "safeobjworld_r_image_only":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.005,
            "area_weight": 0.10,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method == "safeobjworld_r_mask_only":
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.005,
            "area_weight": 0.10,
            "mc_eval": True,
            "eval_k": 5,
        })
        return cfg

    if method in ["object_token_full", "object_token_det", "object_token_no_geom"]:
        cfg.update({
            "stress_train": True,
            "temporal_weight": 0.05,
            "split_weight": 0.0,
            "area_weight": 0.10,
            "mc_eval": method != "object_token_det",
            "eval_k": 5 if method != "object_token_det" else 1,
        })
        return cfg

    return cfg


# ============================================================
# Losses
# ============================================================

def dice_loss_with_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(-1, -2, -3))
    den = probs.sum(dim=(-1, -2, -3)) + target.sum(dim=(-1, -2, -3))
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss_with_logits(logits, target)


def soft_centroid(probs, eps=1e-6):
    b, hh, _, s1, s2 = probs.shape
    device = probs.device
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, s1, device=device),
        torch.linspace(0, 1, s2, device=device),
        indexing="ij",
    )
    mass = probs[:, :, 0].sum(dim=(-1, -2)).clamp_min(eps)
    cx = (probs[:, :, 0] * xx).sum(dim=(-1, -2)) / mass
    cy = (probs[:, :, 0] * yy).sum(dim=(-1, -2)) / mass
    return torch.stack([cx, cy], dim=-1)


def temporal_motion_loss(logits):
    if logits.shape[1] < 3:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)
    cen = soft_centroid(probs)
    accel = cen[:, 2:] - 2 * cen[:, 1:-1] + cen[:, :-2]
    return torch.sqrt((accel ** 2).sum(dim=-1) + 1e-6).mean()


def area_stability_loss(logits, target):
    probs = torch.sigmoid(logits)
    pa = probs.mean(dim=(-1, -2, -3))
    ta = target.mean(dim=(-1, -2, -3))
    area_match = F.smooth_l1_loss(pa, ta)
    if pa.shape[1] >= 2:
        area_smooth = torch.abs(pa[:, 1:] - pa[:, :-1]).mean()
    else:
        area_smooth = pa.sum() * 0.0
    return area_match + 0.2 * area_smooth


def split_proxy_loss(logits):
    """
    Differentiable proxy for fragmented masks.
    It penalizes excessive boundary length relative to soft area.
    This is not connected-components itself, but reduces fragmented/split predictions.
    """
    probs = torch.sigmoid(logits)
    dx = torch.abs(probs[..., :, 1:] - probs[..., :, :-1]).mean()
    dy = torch.abs(probs[..., 1:, :] - probs[..., :-1, :]).mean()
    area = probs.mean().clamp_min(1e-5)
    return (dx + dy) / torch.sqrt(area)


def total_loss(logits, target, cfg):
    loss = bce_dice_loss(logits, target)

    if cfg.get("temporal_weight", 0.0) > 0:
        loss = loss + float(cfg["temporal_weight"]) * temporal_motion_loss(logits)

    if cfg.get("split_weight", 0.0) > 0:
        loss = loss + float(cfg["split_weight"]) * split_proxy_loss(logits)

    if cfg.get("area_weight", 0.0) > 0:
        loss = loss + float(cfg["area_weight"]) * area_stability_loss(logits, target)

    return loss


# ============================================================
# Metrics
# ============================================================

def binary_iou(pred, gt, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + eps))


def boundary_f_score(pred, gt, bound=2):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    if cv2 is None:
        return binary_iou(pred, gt)

    kernel = np.ones((3, 3), np.uint8)
    pred_er = cv2.erode(pred, kernel, iterations=1)
    gt_er = cv2.erode(gt, kernel, iterations=1)

    pred_b = pred ^ pred_er
    gt_b = gt ^ gt_er

    pred_d = cv2.dilate(pred_b, kernel, iterations=bound)
    gt_d = cv2.dilate(gt_b, kernel, iterations=bound)

    precision = (pred_b & gt_d).sum() / max(pred_b.sum(), 1)
    recall = (gt_b & pred_d).sum() / max(gt_b.sum(), 1)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def centroid_np(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return np.nan, np.nan
    h, w = mask.shape
    return float(xs.mean() / max(w - 1, 1)), float(ys.mean() / max(h - 1, 1))


def connected_components_count(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return 0
    if cv2 is None:
        return 1
    n, labels = cv2.connectedComponents(mask)
    return max(0, n - 1)


def compute_mce(centers):
    arr = np.array(centers, dtype=np.float32)
    if len(arr) < 3 or np.isnan(arr).any():
        return np.nan
    accel = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    return float(np.mean(np.sqrt((accel ** 2).sum(axis=1))))


def evaluate_sample_metrics(prob_seq, gt_seq, unc_seq=None, threshold=0.5):
    H = prob_seq.shape[0]
    rows = []
    centers = []

    for h in range(H):
        prob = prob_seq[h]
        pred = prob >= threshold
        gt = gt_seq[h] >= 0.5

        iou = binary_iou(pred, gt)
        bf = boundary_f_score(pred, gt)
        jf = 0.5 * (iou + bf)

        pred_present = pred.sum() > 0
        gt_present = gt.sum() > 0

        vanish = 1.0 if (gt_present and ((not pred_present) or iou < 0.02)) else 0.0
        halluc = 1.0 if ((not gt_present) and pred_present) else 0.0
        ops = 1.0 if ((gt_present and iou >= 0.10) or ((not gt_present) and (not pred_present))) else 0.0

        comps = connected_components_count(pred)
        split = 1.0 if comps > 1 else 0.0

        cx, cy = centroid_np(pred)
        centers.append((cx, cy))

        conf = float(np.mean(np.maximum(prob, 1.0 - prob)))
        unc = float(unc_seq[h]) if unc_seq is not None else 0.0

        rows.append({
            "h": h + 1,
            "iou": iou,
            "boundary_f": bf,
            "jf": jf,
            "ops": ops,
            "vanish": vanish,
            "hallucination": halluc,
            "split": split,
            "components": float(comps),
            "pred_area": float(pred.mean()),
            "gt_area": float(gt.mean()),
            "confidence": conf,
            "uncertainty": unc,
            "correct_iou50": 1.0 if iou >= 0.5 else 0.0,
            "error_1miou": 1.0 - iou,
            "brier_iou50": (conf - (1.0 if iou >= 0.5 else 0.0)) ** 2,
        })

    mce = compute_mce(centers)
    for r in rows:
        r["mce"] = mce

    return rows


def corr_safe(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    if np.std(a[m]) < 1e-8 or np.std(b[m]) < 1e-8:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def ece_binary(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=np.float32)
    correct = np.asarray(correct, dtype=np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(conf)
    if total == 0:
        return np.nan
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            m = (conf >= lo) & (conf < hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / total) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def aurc_from_uncertainty(unc, error):
    unc = np.asarray(unc, dtype=np.float32)
    error = np.asarray(error, dtype=np.float32)
    m = np.isfinite(unc) & np.isfinite(error)
    unc = unc[m]
    error = error[m]
    if len(unc) < 3:
        return np.nan
    order = np.argsort(unc)
    e = error[order]
    risks = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(np.mean(risks))


def selective_mean(metric, unc, keep=0.8):
    metric = np.asarray(metric, dtype=np.float32)
    unc = np.asarray(unc, dtype=np.float32)
    m = np.isfinite(metric) & np.isfinite(unc)
    metric = metric[m]
    unc = unc[m]
    if len(metric) == 0:
        return np.nan
    n = max(1, int(len(metric) * keep))
    order = np.argsort(unc)
    return float(metric[order[:n]].mean())


def summarize_rows(df):
    if len(df) == 0:
        return pd.DataFrame()

    summaries = []
    group_cols = ["method", "stress", "max_h"]

    def add_summary(key, g, h_value):
        method, stress, max_h = key[:3]
        row = {
            "method": method,
            "stress": stress,
            "max_h": max_h,
            "h": h_value,
        }
        for col in [
            "iou", "boundary_f", "jf", "ops", "vanish", "hallucination",
            "split", "components", "mce", "pred_area", "gt_area",
            "uncertainty", "confidence", "brier_iou50"
        ]:
            row[col] = float(g[col].mean(skipna=True))

        row["uec"] = corr_safe(g["uncertainty"], g["error_1miou"])
        row["ece_iou50"] = ece_binary(g["confidence"], g["correct_iou50"])
        row["aurc_unc"] = aurc_from_uncertainty(g["uncertainty"], g["error_1miou"])
        row["selective_jf_80"] = selective_mean(g["jf"], g["uncertainty"], keep=0.8)
        row["n_rows"] = int(len(g))
        row["n_samples"] = int(g["sample_id"].nunique())
        return row

    for key, g in df.groupby(group_cols):
        summaries.append(add_summary(key, g, "all"))

    for key, g in df.groupby(group_cols + ["h"]):
        method, stress, max_h, h = key
        summaries.append(add_summary((method, stress, max_h), g, int(h)))

    return pd.DataFrame(summaries)


# ============================================================
# Baselines
# ============================================================

def shift_mask(mask, dx, dy):
    mask = mask.astype(np.float32)
    if cv2 is None:
        return mask
    h, w = mask.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    out = cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (out > 0.5).astype(np.float32)


def baseline_copy_last(batch):
    x = batch["x_seq"].numpy()
    last = x[:, -1, 3]
    H = batch["target"].shape[1]
    return np.repeat(last[:, None, :, :], H, axis=1)


def baseline_linear_centroid(batch, use_scale=False):
    x = batch["x_seq"].numpy()
    masks = x[:, :, 3]
    B, L, S, _ = masks.shape
    H = batch["target"].shape[1]
    out = np.zeros((B, H, S, S), dtype=np.float32)

    for b in range(B):
        m0 = masks[b, max(0, L - 2)]
        m1 = masks[b, L - 1]

        c0 = centroid_np(m0 > 0.5)
        c1 = centroid_np(m1 > 0.5)

        if np.isnan(c0[0]) or np.isnan(c1[0]):
            dx = dy = 0.0
        else:
            dx = (c1[0] - c0[0]) * (S - 1)
            dy = (c1[1] - c0[1]) * (S - 1)

        cur = m1.copy()
        area0 = max(float((m0 > 0.5).sum()), 1.0)
        area1 = max(float((m1 > 0.5).sum()), 1.0)
        scale_rate = math.sqrt(area1 / area0) if use_scale else 1.0

        for h in range(H):
            cur = shift_mask(cur, dx, dy)
            if use_scale and cv2 is not None:
                bbox = bbox_from_mask(cur)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    patch = cur[y1:y2+1, x1:x2+1]
                    factor = max(0.7, min(1.3, scale_rate))
                    nh = max(1, int(patch.shape[0] * factor))
                    nw = max(1, int(patch.shape[1] * factor))
                    rs = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    canvas = np.zeros_like(cur)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    xx1 = max(0, cx - nw // 2)
                    yy1 = max(0, cy - nh // 2)
                    xx2 = min(S, xx1 + nw)
                    yy2 = min(S, yy1 + nh)
                    canvas[yy1:yy2, xx1:xx2] = rs[:yy2-yy1, :xx2-xx1]
                    cur = canvas
            out[b, h] = cur

    return out


def warp_mask_with_flow(mask, flow):
    if cv2 is None:
        return mask.astype(np.float32)
    h, w = mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 0.5).astype(np.float32)


def baseline_flow_warp(batch, fallback=False):
    x = batch["x_seq"].numpy()
    B, L, C, S, _ = x.shape
    H = batch["target"].shape[1]
    out = np.zeros((B, H, S, S), dtype=np.float32)

    if cv2 is None:
        return baseline_linear_centroid(batch)

    for b in range(B):
        im0 = np.transpose(x[b, max(0, L - 2), :3], (1, 2, 0))
        im1 = np.transpose(x[b, L - 1, :3], (1, 2, 0))
        im0 = (im0 * 255).astype(np.uint8)
        im1 = (im1 * 255).astype(np.uint8)

        g0 = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        cur = x[b, L - 1, 3]

        c0 = centroid_np(x[b, max(0, L - 2), 3] > 0.5)
        c1 = centroid_np(x[b, L - 1, 3] > 0.5)
        if np.isnan(c0[0]) or np.isnan(c1[0]):
            dx = dy = 0.0
        else:
            dx = (c1[0] - c0[0]) * (S - 1)
            dy = (c1[1] - c0[1]) * (S - 1)

        for h in range(H):
            if fallback and h >= 3:
                cur = shift_mask(cur, dx, dy)
            else:
                cur = warp_mask_with_flow(cur, flow)
            out[b, h] = cur

    return out


# ============================================================
# Train/eval
# ============================================================

def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_loader(args, split, stress="clean", random_stress=False, max_samples=0, shuffle=False):
    ds = DAVISObjectFutureDataset(
        dataset_root=args.dataset_root,
        split=split,
        year=args.year,
        prefix_len=args.prefix_len,
        max_h=args.max_h,
        image_size=args.image_size,
        stress=stress,
        random_stress=random_stress,
        max_samples=max_samples,
        seed=args.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        drop_last=False,
    )
    return ds, dl


def train_model(args):
    set_seed(args.seed)

    cfg = method_config(args.method)
    out_dir = ensure_dir(Path(args.run_root) / "models" / args.method / f"H{args.max_h}")
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"
    history_path = out_dir / "history.csv"
    config_path = out_dir / "config.json"

    if args.skip_existing and ckpt_best.exists():
        print(f"[skip train] {ckpt_best}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] method={args.method} device={device} cfg={cfg}")

    _, train_dl = make_loader(
        args,
        split="train",
        stress="clean",
        random_stress=cfg["stress_train"],
        max_samples=args.max_train_samples,
        shuffle=True,
    )

    _, val_dl = make_loader(
        args,
        split="val",
        stress="clean",
        random_stress=False,
        max_samples=min(args.max_val_samples, 512) if args.max_val_samples else 512,
        shuffle=False,
    )

    model = build_model(args.method, args.image_size, args.max_h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_val = float("inf")
    hist = []

    with open(config_path, "w") as f:
        json.dump({"args": vars(args), "method_cfg": cfg}, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                logits = model(batch["x_seq"], batch["geom"], sample=False)
                loss = total_loss(logits, batch["target"], cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"[train:{args.method}:H{args.max_h}] "
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_dl)} "
                    f"loss={np.mean(losses[-20:]):.4f}"
                )

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                batch = batch_to_device(batch, device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                    logits = model(batch["x_seq"], batch["geom"], sample=False)
                    loss = total_loss(logits, batch["target"], cfg)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - t0,
            "method": args.method,
            "max_h": args.max_h,
        }
        hist.append(row)
        pd.DataFrame(hist).to_csv(history_path, index=False)

        ckpt = {
            "model": model.state_dict(),
            "method": args.method,
            "image_size": args.image_size,
            "max_h": args.max_h,
            "method_cfg": cfg,
            "args": vars(args),
        }

        torch.save(ckpt, ckpt_last)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_best)

        print(
            f"[epoch] method={args.method} epoch={epoch} "
            f"train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f}"
        )

    print(f"[done train] {ckpt_best}")


def eval_baseline(args):
    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.run_root) / "eval" / args.method / f"H{args.max_h}" / args.stress)
    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.csv"

    if args.skip_existing and rows_path.exists() and summary_path.exists():
        print(f"[skip baseline] {summary_path}")
        return

    _, dl = make_loader(
        args,
        split="val",
        stress=args.stress,
        random_stress=False,
        max_samples=args.max_val_samples,
        shuffle=False,
    )

    all_rows = []
    t0 = time.time()

    for step, batch in enumerate(dl):
        if args.method == "copy_last":
            pred = baseline_copy_last(batch)
        elif args.method == "linear_centroid":
            pred = baseline_linear_centroid(batch, use_scale=False)
        elif args.method == "linear_centroid_scale":
            pred = baseline_linear_centroid(batch, use_scale=True)
        elif args.method == "flow_warp":
            pred = baseline_flow_warp(batch, fallback=False)
        elif args.method == "flow_linear_fallback":
            pred = baseline_flow_warp(batch, fallback=True)
        else:
            raise ValueError(args.method)

        y = batch["target"].numpy()[:, :, 0]
        B = y.shape[0]

        for b in range(B):
            metrics = evaluate_sample_metrics(pred[b], y[b], unc_seq=None)
            for r in metrics:
                r.update({
                    "method": args.method,
                    "stress": args.stress,
                    "max_h": args.max_h,
                    "seq": batch["seq"][b],
                    "start": int(batch["start"][b]),
                    "oid": int(batch["oid"][b]),
                    "sample_id": batch["sample_id"][b],
                })
                all_rows.append(r)

        if step % 25 == 0:
            print(f"[eval baseline:{args.method}:{args.stress}] step={step}/{len(dl)} rows={len(all_rows)}")

    df = pd.DataFrame(all_rows)
    summary = summarize_rows(df)

    df.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[done baseline] rows={rows_path}")
    print(f"[done baseline] summary={summary_path}")
    print(f"[time] {time.time() - t0:.1f}s")


def eval_model(args):
    set_seed(args.seed)

    cfg = method_config(args.method)
    ckpt_path = Path(args.run_root) / "models" / args.method / f"H{args.max_h}" / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    out_dir = ensure_dir(Path(args.run_root) / "eval" / args.method / f"H{args.max_h}" / args.stress)
    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.csv"

    if args.skip_existing and rows_path.exists() and summary_path.exists():
        print(f"[skip eval] {summary_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[eval model] method={args.method} stress={args.stress} device={device} cfg={cfg}")

    _, dl = make_loader(
        args,
        split="val",
        stress=args.stress,
        random_stress=False,
        max_samples=args.max_val_samples,
        shuffle=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(args.method, args.image_size, args.max_h)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    eval_k = int(cfg.get("eval_k", 1))
    if args.eval_k_override > 0:
        eval_k = int(args.eval_k_override)

    all_rows = []
    t0 = time.time()

    with torch.no_grad():
        for step, batch_cpu in enumerate(dl):
            y_np = batch_cpu["target"].numpy()[:, :, 0]
            batch = batch_to_device(batch_cpu, device)

            samples = []
            for k in range(eval_k):
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                    logits = model(batch["x_seq"], batch["geom"], sample=(eval_k > 1))
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                samples.append(probs)

            stack = np.stack(samples, axis=0)
            mean_probs = stack.mean(axis=0)[:, :, 0]
            unc_map = stack.var(axis=0)[:, :, 0]
            unc_seq = unc_map.mean(axis=(-1, -2))

            B = y_np.shape[0]
            for b in range(B):
                metrics = evaluate_sample_metrics(mean_probs[b], y_np[b], unc_seq=unc_seq[b])
                for r in metrics:
                    r.update({
                        "method": args.method,
                        "stress": args.stress,
                        "max_h": args.max_h,
                        "seq": batch_cpu["seq"][b],
                        "start": int(batch_cpu["start"][b]),
                        "oid": int(batch_cpu["oid"][b]),
                        "sample_id": batch_cpu["sample_id"][b],
                    })
                    all_rows.append(r)

            if step % 25 == 0:
                print(f"[eval model:{args.method}:{args.stress}] step={step}/{len(dl)} rows={len(all_rows)}")

    df = pd.DataFrame(all_rows)
    summary = summarize_rows(df)

    df.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[done eval] rows={rows_path}")
    print(f"[done eval] summary={summary_path}")
    print(f"[time] {time.time() - t0:.1f}s")


# ============================================================
# Manifest / aggregate / paper tables
# ============================================================

def make_manifest(args):
    davis_root = locate_davis_root(args.dataset_root)
    out_dir = ensure_dir(args.run_root)
    manifest_path = out_dir / "manifest.json"

    info = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "davis_root": str(davis_root),
        "year": args.year,
        "prefix_len": args.prefix_len,
        "max_h": args.max_h,
        "image_size": args.image_size,
        "splits": {},
    }

    for split in ["train", "val"]:
        seqs = read_split(davis_root, args.year, split)
        ds = DAVISObjectFutureDataset(
            dataset_root=args.dataset_root,
            split=split,
            year=args.year,
            prefix_len=args.prefix_len,
            max_h=args.max_h,
            image_size=args.image_size,
            max_samples=0,
            seed=args.seed,
        )
        info["splits"][split] = {
            "n_sequences": len(seqs),
            "n_samples": len(ds),
            "sequences": seqs,
        }
        print(f"[manifest] {split}: sequences={len(seqs)} samples={len(ds)}")

    with open(manifest_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[manifest] saved {manifest_path}")


def make_stress_drop_table(df_all):
    clean = df_all[df_all["stress"] == "clean"].copy()
    rows = []

    for _, base in clean.iterrows():
        method = base["method"]
        max_h = base["max_h"]

        for stress in sorted(df_all["stress"].unique()):
            if stress == "clean":
                continue

            g = df_all[
                (df_all["method"] == method) &
                (df_all["max_h"] == max_h) &
                (df_all["stress"] == stress)
            ]

            if len(g) == 0:
                continue

            s = g.iloc[0]
            rows.append({
                "method": method,
                "max_h": max_h,
                "stress": stress,
                "jf_clean": base["jf"],
                "jf_stress": s["jf"],
                "jf_drop": base["jf"] - s["jf"],
                "ops_clean": base["ops"],
                "ops_stress": s["ops"],
                "ops_drop": base["ops"] - s["ops"],
                "split_clean": base["split"],
                "split_stress": s["split"],
                "split_increase": s["split"] - base["split"],
                "vanish_clean": base["vanish"],
                "vanish_stress": s["vanish"],
                "vanish_increase": s["vanish"] - base["vanish"],
            })

    return pd.DataFrame(rows)


def aggregate(args):
    run_root = Path(args.run_root)
    out_dir = ensure_dir(run_root / "RESULTS")

    summary_paths = sorted((run_root / "eval").rglob("summary.csv"))
    row_paths = sorted((run_root / "eval").rglob("rows.csv"))

    if not summary_paths:
        print("[aggregate] no summaries found")
        return

    df_sum = pd.concat([pd.read_csv(p) for p in summary_paths], ignore_index=True)
    df_sum.to_csv(out_dir / "all_summaries.csv", index=False)

    df_all = df_sum[df_sum["h"].astype(str) == "all"].copy()
    df_h = df_sum[df_sum["h"].astype(str) != "all"].copy()

    if len(df_all):
        cols_main = [
            "method", "stress", "max_h",
            "iou", "boundary_f", "jf",
            "ops", "vanish", "hallucination", "split", "mce",
            "uncertainty", "uec", "ece_iou50", "brier_iou50",
            "aurc_unc", "selective_jf_80", "n_samples"
        ]
        cols_main = [c for c in cols_main if c in df_all.columns]
        df_all[cols_main].to_csv(out_dir / "table_1_main_all_methods.csv", index=False)

        clean = df_all[df_all["stress"] == "clean"].copy()
        if len(clean):
            clean[cols_main].to_csv(out_dir / "table_2_clean_reliability.csv", index=False)

        stress_drop = make_stress_drop_table(df_all)
        stress_drop.to_csv(out_dir / "table_3_stress_drop.csv", index=False)

    if len(df_h):
        cols_h = [
            "method", "stress", "max_h", "h",
            "iou", "boundary_f", "jf", "ops",
            "vanish", "hallucination", "split", "mce",
            "uncertainty", "uec", "ece_iou50", "n_samples"
        ]
        cols_h = [c for c in cols_h if c in df_h.columns]
        df_h[cols_h].to_csv(out_dir / "table_4_horizon_breakdown.csv", index=False)

    if row_paths:
        df_rows = pd.concat([pd.read_csv(p) for p in row_paths], ignore_index=True)
        df_rows.to_csv(out_dir / "all_rows.csv", index=False)

    # Compact tables by experimental block.
    if len(df_all):
        method_order = [
            "copy_last",
            "linear_centroid",
            "linear_centroid_scale",
            "flow_warp",
            "flow_linear_fallback",
            "convlstm_base",
            "convlstm_stress",
            "safeobjworld_r_full",
            "safeobjworld_r_no_temporal",
            "safeobjworld_r_no_split",
            "safeobjworld_r_no_area",
            "safeobjworld_r_no_stress",
            "safeobjworld_r_det",
            "safeobjworld_r_image_only",
            "safeobjworld_r_mask_only",
            "object_token_full",
            "object_token_det",
            "object_token_no_geom",
        ]

        df_all["method"] = pd.Categorical(df_all["method"], categories=method_order, ordered=True)
        df_all = df_all.sort_values(["stress", "method"])

        # Experiment 1: standard prediction
        exp1 = df_all[df_all["stress"] == "clean"].copy()
        exp1_cols = ["method", "jf", "iou", "boundary_f", "ops", "split", "mce"]
        exp1_cols = [c for c in exp1_cols if c in exp1.columns]
        exp1[exp1_cols].to_csv(out_dir / "EXP1_standard_prediction_clean.csv", index=False)

        # Experiment 2: reliability audit
        exp2_cols = ["method", "stress", "jf", "ops", "vanish", "hallucination", "split", "mce"]
        exp2_cols = [c for c in exp2_cols if c in df_all.columns]
        df_all[exp2_cols].to_csv(out_dir / "EXP2_reliability_audit_all_stress.csv", index=False)

        # Experiment 3: stress robustness
        if (out_dir / "table_3_stress_drop.csv").exists():
            pd.read_csv(out_dir / "table_3_stress_drop.csv").to_csv(
                out_dir / "EXP3_stress_robustness.csv",
                index=False,
            )

        # Experiment 5: uncertainty
        exp5_methods = [
            "safeobjworld_r_full",
            "safeobjworld_r_det",
            "safeobjworld_r_no_stress",
            "object_token_full",
            "object_token_det",
        ]
        exp5 = df_all[df_all["method"].astype(str).isin(exp5_methods)].copy()
        exp5_cols = [
            "method", "stress", "uncertainty", "uec",
            "ece_iou50", "brier_iou50", "aurc_unc", "selective_jf_80"
        ]
        exp5_cols = [c for c in exp5_cols if c in exp5.columns]
        exp5[exp5_cols].to_csv(out_dir / "EXP5_uncertainty_calibration.csv", index=False)

        # Experiment 6: ablations
        abl_methods = [
            "safeobjworld_r_full",
            "safeobjworld_r_no_temporal",
            "safeobjworld_r_no_split",
            "safeobjworld_r_no_area",
            "safeobjworld_r_no_stress",
            "safeobjworld_r_det",
            "safeobjworld_r_image_only",
            "safeobjworld_r_mask_only",
            "object_token_full",
            "object_token_det",
            "object_token_no_geom",
        ]
        exp6 = df_all[df_all["method"].astype(str).isin(abl_methods)].copy()
        exp6_cols = [
            "method", "stress", "jf", "ops", "vanish",
            "hallucination", "split", "mce", "uec", "ece_iou50"
        ]
        exp6_cols = [c for c in exp6_cols if c in exp6.columns]
        exp6[exp6_cols].to_csv(out_dir / "EXP6_ablation_study.csv", index=False)

    print(f"[aggregate] saved to {out_dir}")
    for p in sorted(out_dir.glob("*.csv")):
        print(f"  {p.name}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", required=True, choices=[
        "make_manifest",
        "train",
        "eval_model",
        "eval_baseline",
        "aggregate",
    ])

    p.add_argument("--dataset_root", type=str, default=str(Path.home() / "SafeObjWorld" / "data" / "DAVIS2017"))
    p.add_argument("--run_root", type=str, default=str(Path.home() / "SafeObjWorld" / "runs_v2_full"))

    p.add_argument("--year", type=str, default="2017")
    p.add_argument("--method", type=str, default="safeobjworld_r_full")
    p.add_argument("--stress", type=str, default="clean")

    p.add_argument("--prefix_len", type=int, default=3)
    p.add_argument("--max_h", type=int, default=5)
    p.add_argument("--image_size", type=int, default=192)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval_k_override", type=int, default=0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--log_every", type=int, default=25)

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 100)
    print("SafeObjWorld V2 runner")
    print(json.dumps(vars(args), indent=2))
    print("=" * 100)

    if args.mode == "make_manifest":
        make_manifest(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "eval_baseline":
        eval_baseline(args)
    elif args.mode == "eval_model":
        eval_model(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
PY

# ============================================================
# Dynamic scheduler
# ============================================================

cat > "$CODE_DIR/launch_safeobjworld_v2.py" <<'PY'
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Job:
    name: str
    cmd: list
    log_path: Path


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_capture(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""


def query_gpus():
    out = run_capture([
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    if not out:
        return gpus

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "free_mb": int(parts[1]),
                "total_mb": int(parts[2]),
                "util": int(parts[3]),
            })
        except Exception:
            pass
    return gpus


def launch(job, gpu_id=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ensure_dir(job.log_path.parent)
    f = open(job.log_path, "w", buffering=1)

    print(f"[launch] {job.name}")
    print(f"         gpu={gpu_id}")
    print(f"         log={job.log_path}")
    print("         cmd=" + " ".join(shlex.quote(str(x)) for x in job.cmd))

    proc = subprocess.Popen(
        job.cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return proc, f


def run_scheduler(
    jobs,
    min_free_mb=9000,
    max_jobs_per_gpu=2,
    poll_seconds=20,
    start_gap_seconds=10,
    cpu_parallel=2,
):
    pending = list(jobs)
    running = []

    print("=" * 100)
    print(f"Scheduler start. jobs={len(pending)}")
    print(f"min_free_mb={min_free_mb} max_jobs_per_gpu={max_jobs_per_gpu}")
    print("=" * 100)

    while pending or running:
        still = []
        for proc, log_f, job, gpu in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, log_f, job, gpu))
            else:
                log_f.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[done] {job.name} gpu={gpu} status={status}")
                if ret != 0:
                    print(f"       log={job.log_path}")
        running = still

        gpus = query_gpus()

        if not gpus:
            running_cpu = sum(1 for _, _, _, gpu in running if gpu is None)
            while pending and running_cpu < cpu_parallel:
                job = pending.pop(0)
                proc, log_f = launch(job, None)
                running.append((proc, log_f, job, None))
                running_cpu += 1
                time.sleep(start_gap_seconds)
            time.sleep(poll_seconds)
            continue

        running_per_gpu = {g["index"]: 0 for g in gpus}
        for _, _, _, gpu in running:
            if gpu is not None and gpu in running_per_gpu:
                running_per_gpu[gpu] += 1

        gpus = sorted(gpus, key=lambda x: x["free_mb"], reverse=True)
        launched_any = False

        for g in gpus:
            if not pending:
                break

            gid = g["index"]

            if running_per_gpu.get(gid, 0) >= max_jobs_per_gpu:
                continue

            if g["free_mb"] < min_free_mb:
                continue

            job = pending.pop(0)
            proc, log_f = launch(job, gid)
            running.append((proc, log_f, job, gid))
            running_per_gpu[gid] = running_per_gpu.get(gid, 0) + 1
            launched_any = True
            time.sleep(start_gap_seconds)

        if not launched_any:
            msg = " | ".join([f"gpu{g['index']}:free={g['free_mb']}MB util={g['util']}%" for g in gpus])
            print(f"[wait] pending={len(pending)} running={len(running)} :: {msg}")

        time.sleep(poll_seconds)

    print("=" * 100)
    print("Scheduler finished.")
    print("=" * 100)


def maybe_add(jobs, name, cmd, log_dir, skip_path=None, force=False):
    if skip_path is not None and Path(skip_path).exists() and not force:
        print(f"[skip existing] {name} -> {skip_path}")
        return
    jobs.append(Job(name=name, cmd=cmd, log_path=Path(log_dir) / f"{name}.log"))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", type=str, default=str(Path.home() / "SafeObjWorld"))
    ap.add_argument("--dataset_root", type=str, default=str(Path.home() / "SafeObjWorld" / "data" / "DAVIS2017"))
    ap.add_argument("--run_root", type=str, default=str(Path.home() / "SafeObjWorld" / "runs_v2_full"))

    ap.add_argument("--image_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--prefix_len", type=int, default=3)
    ap.add_argument("--max_h", type=int, default=5)

    ap.add_argument("--run_h10", type=int, default=0)
    ap.add_argument("--run_optional_stress", type=int, default=0)

    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--min_free_mb", type=int, default=9000)
    ap.add_argument("--max_jobs_per_gpu", type=int, default=2)
    ap.add_argument("--poll_seconds", type=int, default=20)
    ap.add_argument("--start_gap_seconds", type=int, default=10)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--eval_only", action="store_true")

    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    logs = ensure_dir(run_root / "logs")
    runner = project_root / "code" / "safeobjworld_v2.py"

    print("=" * 100)
    print("SafeObjWorld V2 launcher")
    print(json.dumps(vars(args), indent=2))
    print("=" * 100)

    base_common = [
        "--dataset_root", str(dataset_root),
        "--run_root", str(run_root),
        "--prefix_len", str(args.prefix_len),
        "--image_size", str(args.image_size),
        "--batch_size", str(args.batch_size),
        "--max_train_samples", str(args.max_train_samples),
        "--max_val_samples", str(args.max_val_samples),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--skip_existing",
    ]
    if args.amp:
        base_common.append("--amp")

    max_h_values = [args.max_h]
    if args.run_h10:
        max_h_values.append(10)

    # --------------------------------------------------------
    # Phase 0: manifest
    # --------------------------------------------------------
    if not args.eval_only:
        cmd = [
            sys.executable, str(runner),
            "--mode", "make_manifest",
            "--dataset_root", str(dataset_root),
            "--run_root", str(run_root),
            "--prefix_len", str(args.prefix_len),
            "--max_h", str(args.max_h),
            "--image_size", str(args.image_size),
            "--seed", str(args.seed),
        ]
        subprocess.run(cmd, check=True)

    # --------------------------------------------------------
    # Phase 1: train learned methods
    # --------------------------------------------------------
    learned_methods = [
        # learned baseline
        "convlstm_base",
        "convlstm_stress",

        # proposed method + core ablations
        "safeobjworld_r_full",
        "safeobjworld_r_no_temporal",
        "safeobjworld_r_no_split",
        "safeobjworld_r_no_area",
        "safeobjworld_r_no_stress",
        "safeobjworld_r_det",
        "safeobjworld_r_image_only",
        "safeobjworld_r_mask_only",

        # architecture ablations from old idea
        "object_token_full",
        "object_token_det",
        "object_token_no_geom",
    ]

    train_jobs = []

    if not args.eval_only:
        for H in max_h_values:
            for method in learned_methods:
                ckpt = run_root / "models" / method / f"H{H}" / "best.pt"

                cmd = [
                    sys.executable, str(runner),
                    "--mode", "train",
                    "--method", method,
                    "--max_h", str(H),
                    "--epochs", str(args.epochs),
                    *base_common,
                ]

                maybe_add(
                    train_jobs,
                    name=f"train_{method}_H{H}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=ckpt,
                    force=args.force,
                )

        print(f"[phase 1] train jobs={len(train_jobs)}")
        run_scheduler(
            train_jobs,
            min_free_mb=args.min_free_mb,
            max_jobs_per_gpu=args.max_jobs_per_gpu,
            poll_seconds=args.poll_seconds,
            start_gap_seconds=args.start_gap_seconds,
        )

    if args.train_only:
        print("[train_only] stopping after training")
        return

    # --------------------------------------------------------
    # Phase 2: eval baselines + learned methods
    # --------------------------------------------------------
    baseline_methods = [
        "copy_last",
        "linear_centroid",
        "linear_centroid_scale",
        "flow_warp",
        "flow_linear_fallback",
    ]

    stresses = ["clean", "occlusion", "blur", "frame_drop"]
    if args.run_optional_stress:
        stresses += ["lowlight", "noise", "distractor"]

    eval_jobs = []

    for H in max_h_values:
        for stress in stresses:
            for method in baseline_methods:
                summary = run_root / "eval" / method / f"H{H}" / stress / "summary.csv"
                cmd = [
                    sys.executable, str(runner),
                    "--mode", "eval_baseline",
                    "--method", method,
                    "--stress", stress,
                    "--max_h", str(H),
                    *base_common,
                ]
                maybe_add(
                    eval_jobs,
                    name=f"eval_{method}_H{H}_{stress}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=summary,
                    force=args.force,
                )

            for method in learned_methods:
                summary = run_root / "eval" / method / f"H{H}" / stress / "summary.csv"
                cmd = [
                    sys.executable, str(runner),
                    "--mode", "eval_model",
                    "--method", method,
                    "--stress", stress,
                    "--max_h", str(H),
                    *base_common,
                ]
                maybe_add(
                    eval_jobs,
                    name=f"eval_{method}_H{H}_{stress}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=summary,
                    force=args.force,
                )

    print(f"[phase 2] eval jobs={len(eval_jobs)}")
    run_scheduler(
        eval_jobs,
        min_free_mb=args.min_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        poll_seconds=args.poll_seconds,
        start_gap_seconds=args.start_gap_seconds,
    )

    # --------------------------------------------------------
    # Phase 3: aggregate
    # --------------------------------------------------------
    cmd = [
        sys.executable, str(runner),
        "--mode", "aggregate",
        "--dataset_root", str(dataset_root),
        "--run_root", str(run_root),
        "--prefix_len", str(args.prefix_len),
        "--max_h", str(args.max_h),
        "--image_size", str(args.image_size),
        "--seed", str(args.seed),
    ]
    subprocess.run(cmd, check=True)

    print("=" * 100)
    print("DONE")
    print(f"RUN_ROOT={run_root}")
    print(f"RESULTS={run_root / 'RESULTS'}")
    print(f"LOGS={logs}")
    print("=" * 100)


if __name__ == "__main__":
    main()
PY

# ============================================================
# Run wrapper
# ============================================================

cat > "$PROJECT_ROOT/run_v2_full.sh" <<'BASH_RUN'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$HOME/SafeObjWorld"
VENV_DIR="$PROJECT_ROOT/.venv_safeobjworld"
DATA_ROOT="$PROJECT_ROOT/data/DAVIS2017"

source "$VENV_DIR/bin/activate"

if [ "${KILL_OLD:-0}" = "1" ]; then
  pkill -f "safeobjworld_v2.py" || true
  pkill -f "launch_safeobjworld_v2.py" || true
fi

if [ "${QUICK:-0}" = "1" ]; then
  RUN_ROOT="$PROJECT_ROOT/runs_v2_smoke"
  IMAGE_SIZE=160
  BATCH_SIZE=8
  EPOCHS=2
  MAX_TRAIN_SAMPLES=512
  MAX_VAL_SAMPLES=256
  MAX_JOBS_PER_GPU=1
  MIN_FREE_MB=7000
  RUN_H10=0
  OPTIONAL_STRESS=0
else
  RUN_ROOT="$PROJECT_ROOT/runs_v2_full"
  IMAGE_SIZE="${IMAGE_SIZE:-192}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  EPOCHS="${EPOCHS:-10}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
  MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
  MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
  MIN_FREE_MB="${MIN_FREE_MB:-9000}"
  RUN_H10="${RUN_H10:-0}"
  OPTIONAL_STRESS="${OPTIONAL_STRESS:-0}"
fi

if [ "${CLEAN:-0}" = "1" ]; then
  echo "[CLEAN=1] removing $RUN_ROOT"
  rm -rf "$RUN_ROOT"
fi

mkdir -p "$RUN_ROOT/logs"

echo "============================================================"
echo "SafeObjWorld V2 launch"
echo "RUN_ROOT=$RUN_ROOT"
echo "IMAGE_SIZE=$IMAGE_SIZE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES"
echo "MAX_VAL_SAMPLES=$MAX_VAL_SAMPLES"
echo "MAX_JOBS_PER_GPU=$MAX_JOBS_PER_GPU"
echo "MIN_FREE_MB=$MIN_FREE_MB"
echo "RUN_H10=$RUN_H10"
echo "OPTIONAL_STRESS=$OPTIONAL_STRESS"
echo "============================================================"

python "$PROJECT_ROOT/code/launch_safeobjworld_v2.py" \
  --project_root "$PROJECT_ROOT" \
  --dataset_root "$DATA_ROOT" \
  --run_root "$RUN_ROOT" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --prefix_len 3 \
  --max_h 5 \
  --run_h10 "$RUN_H10" \
  --run_optional_stress "$OPTIONAL_STRESS" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --num_workers 4 \
  --max_jobs_per_gpu "$MAX_JOBS_PER_GPU" \
  --min_free_mb "$MIN_FREE_MB" \
  --poll_seconds 20 \
  --start_gap_seconds 10 \
  --amp

echo "============================================================"
echo "FINISHED"
echo "Main results:"
echo "$RUN_ROOT/RESULTS"
echo "============================================================"
BASH_RUN

chmod +x "$PROJECT_ROOT/run_v2_full.sh"

echo "============================================================"
echo "V2 code is ready."
echo "Runner:   $CODE_DIR/safeobjworld_v2.py"
echo "Launcher: $CODE_DIR/launch_safeobjworld_v2.py"
echo "Wrapper:  $PROJECT_ROOT/run_v2_full.sh"
echo "============================================================"
