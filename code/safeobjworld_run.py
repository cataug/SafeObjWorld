
import argparse
import json
import math
import os
import random
import time
import warnings
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
    # Prevent CPU oneDNN/MKLDNN primitive crashes if CUDA is unavailable.
    torch.backends.mkldnn.enabled = False
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------
# General utils
# ------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png"}


def set_seed(seed: int):
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
    raise FileNotFoundError(
        f"Could not locate DAVIS root under {dataset_root}. "
        "Expected either <root>/DAVIS or <root>/JPEGImages + <root>/Annotations."
    )


def read_split(davis_root, year="2017", split="train"):
    p = Path(davis_root) / "ImageSets" / str(year) / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
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


def mask_geom(mask):
    mask = mask.astype(bool)
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(6, dtype=np.float32)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = xs.mean() / max(w - 1, 1)
    cy = ys.mean() / max(h - 1, 1)
    area = float(mask.mean())
    return np.array([
        cx,
        cy,
        x1 / max(w - 1, 1),
        y1 / max(h - 1, 1),
        x2 / max(w - 1, 1),
        y2 / max(h - 1, 1),
    ], dtype=np.float32) * np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)


def bbox_from_mask(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def apply_occlusion_to_prefix(images, masks, rng, severity=0.45):
    # Occlude the current object region in the last prefix frame and a related region
    # in previous frames. This tests object permanence under partial visibility.
    out_images = [im.copy() for im in images]
    out_masks = [m.copy() for m in masks]

    ref = out_masks[-1]
    bbox = bbox_from_mask(ref)
    if bbox is None:
        return out_images, out_masks

    x1, y1, x2, y2 = bbox
    h, w = ref.shape
    bw = max(2, x2 - x1 + 1)
    bh = max(2, y2 - y1 + 1)

    occ_w = max(2, int(bw * severity))
    occ_h = max(2, int(bh * severity))

    ox1 = int(np.clip(x1 + rng.uniform(0.1, 0.5) * bw, 0, w - 1))
    oy1 = int(np.clip(y1 + rng.uniform(0.1, 0.5) * bh, 0, h - 1))
    ox2 = int(np.clip(ox1 + occ_w, 0, w))
    oy2 = int(np.clip(oy1 + occ_h, 0, h))

    for i in range(len(out_images)):
        arr = np.array(out_images[i]).copy()
        arr[oy1:oy2, ox1:ox2, :] = 0
        out_images[i] = Image.fromarray(arr.astype(np.uint8))
        out_masks[i][oy1:oy2, ox1:ox2] = 0.0

    return out_images, out_masks


def apply_stress(images, masks, stress, rng):
    if stress in [None, "clean"]:
        return images, masks

    images = [im.copy() for im in images]
    masks = [m.copy() for m in masks]

    if stress == "blur":
        return [im.filter(ImageFilter.GaussianBlur(radius=2.0)) for im in images], masks

    if stress == "lowlight":
        return [ImageEnhance.Brightness(im).enhance(0.45) for im in images], masks

    if stress == "occlusion":
        return apply_occlusion_to_prefix(images, masks, rng=rng, severity=0.45)

    if stress == "frame_drop":
        # Replace the middle prefix frame by the first one.
        # Keeps tensor shape fixed but removes temporal evidence.
        if len(images) >= 3:
            mid = len(images) // 2
            images[mid] = images[0].copy()
            masks[mid] = masks[0].copy()
        return images, masks

    if stress == "noise":
        out = []
        for im in images:
            arr = np.array(im).astype(np.float32)
            noise = rng.normal(0, 18, arr.shape)
            arr = np.clip(arr + noise, 0, 255)
            out.append(Image.fromarray(arr.astype(np.uint8)))
        return out, masks

    raise ValueError(f"Unknown stress: {stress}")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class DAVISObjectFutureDataset(Dataset):
    """
    Per-object future mask prediction dataset.

    Each sample:
      input prefix: L RGB frames + L binary masks for one object id
      target: H future binary masks for same object id
    """

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
            masks = sorted_files(ann_dir, MASK_EXTS)
            stem_to_img = {p.stem: p for p in imgs}
            stem_to_ann = {p.stem: p for p in masks}
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
            raise RuntimeError(
                f"No samples created. root={self.dataset_root}, split={split}, "
                f"prefix_len={prefix_len}, max_h={max_h}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_object_mask(self, ann_path, oid):
        label = load_label_mask(ann_path)
        return (label == int(oid)).astype(np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s["seq"]
        start = s["start"]
        oid = int(s["oid"])
        img_paths = s["img_paths"]
        ann_paths = s["ann_paths"]

        prefix_images = []
        prefix_masks = []
        for j in range(start, start + self.prefix_len):
            prefix_images.append(load_rgb(img_paths[j]))
            prefix_masks.append(self._load_object_mask(ann_paths[j], oid))

        target_masks = []
        for h in range(1, self.max_h + 1):
            j = start + self.prefix_len - 1 + h
            target_masks.append(self._load_object_mask(ann_paths[j], oid))

        rng = np.random.default_rng(self.seed + idx)

        if self.random_stress:
            stress = rng.choice(["clean", "blur", "occlusion", "frame_drop"])
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

        x_seq = torch.tensor(np.stack(x_seq, axis=0), dtype=torch.float32)       # L,4,S,S
        geom_seq = torch.tensor(np.stack(geom_seq, axis=0), dtype=torch.float32) # L,6
        y = torch.tensor(np.stack(y, axis=0), dtype=torch.float32)               # H,1,S,S

        return {
            "x_seq": x_seq,
            "geom": geom_seq,
            "target": y,
            "seq": seq,
            "start": int(start),
            "oid": int(oid),
            "stress": str(stress),
            "sample_id": f"{seq}:{start}:{oid}",
        }


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

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


class ConvLSTMFuture(nn.Module):
    def __init__(self, in_ch=4, hidden=64, max_h=5):
        super().__init__()
        self.max_h = max_h
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.cell = ConvLSTMCell(hidden, hidden)
        self.roll_cell = ConvLSTMCell(hidden, hidden)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden, 48, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 3, padding=1),
        )

    def forward(self, x_seq, geom=None, sample=False):
        # x_seq: B,L,4,S,S
        b, l, c, h, w = x_seq.shape
        h0 = w0 = None
        state = None
        last_feat = None

        for t in range(l):
            feat = self.enc(x_seq[:, t])
            last_feat = feat
            if state is None:
                zeros = torch.zeros(
                    b, feat.shape[1], feat.shape[2], feat.shape[3],
                    device=feat.device, dtype=feat.dtype
                )
                state = (zeros, zeros)
            state = self.cell(feat, state)

        outs = []
        inp = last_feat
        for _ in range(self.max_h):
            state = self.roll_cell(inp, state)
            logits = self.dec(state[0])
            outs.append(logits)
            inp = torch.zeros_like(inp)

        return torch.stack(outs, dim=1)  # B,H,1,S,S


class SafeObjWorldNet(nn.Module):
    """
    Lightweight object-token latent dynamics.
    Encodes each timestep as an object token from image+mask and optional geometry,
    rolls latent state forward, decodes future masks.
    """

    def __init__(
        self,
        image_size=192,
        in_ch=4,
        geom_dim=6,
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
            tok = self.token_proj(tok)
            tokens.append(tok)
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
        tokens = self.encode_tokens(x_seq, geom)
        _, h = self.gru(tokens)
        z = h[-1]

        outs = []
        inp = self.step_token.repeat(z.shape[0], 1)

        for _ in range(self.max_h):
            z = self.roll(inp, z)

            if self.stochastic and (self.training or sample):
                z_dec = self.drop(z)
                z_dec = z_dec + 0.03 * torch.randn_like(z_dec)
            else:
                z_dec = z

            outs.append(self.decode_mask(z_dec))
            inp = self.step_token.repeat(z.shape[0], 1)

        return torch.stack(outs, dim=1)


def build_model(method, image_size, max_h):
    if method == "convlstm":
        return ConvLSTMFuture(in_ch=4, hidden=64, max_h=max_h)

    if method == "safeobj_full":
        return SafeObjWorldNet(
            image_size=image_size,
            max_h=max_h,
            use_geom=True,
            stochastic=True,
            dropout=0.15,
        )

    if method == "safeobj_det":
        return SafeObjWorldNet(
            image_size=image_size,
            max_h=max_h,
            use_geom=True,
            stochastic=False,
            dropout=0.0,
        )

    if method == "safeobj_no_geom":
        return SafeObjWorldNet(
            image_size=image_size,
            max_h=max_h,
            use_geom=False,
            stochastic=True,
            dropout=0.15,
        )

    if method == "safeobj_no_stress":
        return SafeObjWorldNet(
            image_size=image_size,
            max_h=max_h,
            use_geom=True,
            stochastic=True,
            dropout=0.15,
        )

    raise ValueError(f"Unknown model method: {method}")


# ------------------------------------------------------------
# Losses
# ------------------------------------------------------------

def dice_loss_with_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_with_logits(logits, target)
    return bce + dice


def soft_centroid(probs, eps=1e-6):
    # probs: B,H,1,S,S
    b, hh, _, s1, s2 = probs.shape
    device = probs.device
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, s1, device=device),
        torch.linspace(0, 1, s2, device=device),
        indexing="ij",
    )
    mass = probs.sum(dim=(-1, -2)).clamp_min(eps)  # B,H,1
    cx = (probs[:, :, 0] * xx).sum(dim=(-1, -2)) / mass[:, :, 0]
    cy = (probs[:, :, 0] * yy).sum(dim=(-1, -2)) / mass[:, :, 0]
    return torch.stack([cx, cy], dim=-1)  # B,H,2


def temporal_motion_loss(logits):
    if logits.shape[1] < 3:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)
    cen = soft_centroid(probs)
    accel = cen[:, 2:] - 2 * cen[:, 1:-1] + cen[:, :-2]
    return (accel ** 2).mean()


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

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


def compute_sequence_mce(centers):
    # centers: H x 2, may contain nan
    arr = np.array(centers, dtype=np.float32)
    if len(arr) < 3 or np.isnan(arr).any():
        return np.nan
    accel = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    return float(np.mean(np.sqrt((accel ** 2).sum(axis=1))))


def evaluate_sample_metrics(prob_seq, gt_seq, unc_seq=None, threshold=0.5):
    # prob_seq: H,S,S
    # gt_seq: H,S,S
    H = prob_seq.shape[0]
    pred_centers = []
    rows = []

    for h in range(H):
        prob = prob_seq[h]
        pred = prob >= threshold
        gt = gt_seq[h] >= 0.5

        iou = binary_iou(pred, gt)
        bf = boundary_f_score(pred, gt)
        jf = 0.5 * (iou + bf)

        pred_area = float(pred.mean())
        gt_area = float(gt.mean())

        gt_present = gt.sum() > 0
        pred_present = pred.sum() > 0

        vanish = 1.0 if (gt_present and (not pred_present or iou < 0.02)) else 0.0
        halluc = 1.0 if ((not gt_present) and pred_present) else 0.0
        ops = 1.0 if ((gt_present and iou >= 0.10) or ((not gt_present) and (not pred_present))) else 0.0

        n_comp = connected_components_count(pred)
        split = 1.0 if n_comp > 1 else 0.0

        cx, cy = centroid_np(pred)
        pred_centers.append((cx, cy))

        conf = float(np.mean(np.maximum(prob, 1 - prob)))
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
            "pred_area": pred_area,
            "gt_area": gt_area,
            "components": float(n_comp),
            "cx_pred": cx,
            "cy_pred": cy,
            "uncertainty": unc,
            "confidence": conf,
            "correct_iou50": 1.0 if iou >= 0.5 else 0.0,
            "error_1miou": 1.0 - iou,
        })

    mce = compute_sequence_mce(pred_centers)
    for r in rows:
        r["mce"] = mce

    return rows


def summarize_rows(df):
    if len(df) == 0:
        return pd.DataFrame()

    summaries = []

    def corr_safe(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        if np.std(a[mask]) < 1e-8 or np.std(b[mask]) < 1e-8:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

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
            m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
            if m.sum() == 0:
                continue
            ece += (m.sum() / total) * abs(correct[m].mean() - conf[m].mean())
        return float(ece)

    group_cols = ["method", "stress", "max_h"]
    for key, g in df.groupby(group_cols):
        row = {k: v for k, v in zip(group_cols, key)}
        row["h"] = "all"
        for col in ["iou", "boundary_f", "jf", "ops", "vanish", "hallucination", "split", "mce", "uncertainty", "confidence"]:
            row[col] = float(g[col].mean(skipna=True))
        row["uec"] = corr_safe(g["uncertainty"], g["error_1miou"])
        row["ece_iou50"] = ece_binary(g["confidence"], g["correct_iou50"])
        row["n_rows"] = int(len(g))
        row["n_samples"] = int(g["sample_id"].nunique())
        summaries.append(row)

    for key, g in df.groupby(group_cols + ["h"]):
        method, stress, max_h, h = key
        row = {"method": method, "stress": stress, "max_h": max_h, "h": int(h)}
        for col in ["iou", "boundary_f", "jf", "ops", "vanish", "hallucination", "split", "mce", "uncertainty", "confidence"]:
            row[col] = float(g[col].mean(skipna=True))
        row["uec"] = corr_safe(g["uncertainty"], g["error_1miou"])
        row["ece_iou50"] = ece_binary(g["confidence"], g["correct_iou50"])
        row["n_rows"] = int(len(g))
        row["n_samples"] = int(g["sample_id"].nunique())
        summaries.append(row)

    return pd.DataFrame(summaries)


# ------------------------------------------------------------
# Baselines
# ------------------------------------------------------------

def shift_mask(mask, dx, dy):
    mask = mask.astype(np.float32)
    if cv2 is None:
        return mask
    h, w = mask.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    out = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (out > 0.5).astype(np.float32)


def baseline_copy_last(batch):
    x = batch["x_seq"]  # B,L,4,S,S
    last = x[:, -1, 3].numpy()
    B, S, _ = last.shape
    H = batch["target"].shape[1]
    return np.repeat(last[:, None, :, :], H, axis=1)


def baseline_linear_centroid(batch):
    x = batch["x_seq"]
    masks = x[:, :, 3].numpy()
    B, L, S, _ = masks.shape
    H = batch["target"].shape[1]
    out = np.zeros((B, H, S, S), dtype=np.float32)

    for b in range(B):
        c0 = centroid_np(masks[b, max(0, L - 2)] > 0.5)
        c1 = centroid_np(masks[b, L - 1] > 0.5)
        if np.isnan(c0[0]) or np.isnan(c1[0]):
            dx = dy = 0.0
        else:
            dx = (c1[0] - c0[0]) * (S - 1)
            dy = (c1[1] - c0[1]) * (S - 1)

        cur = masks[b, L - 1]
        for h in range(H):
            cur = shift_mask(cur, dx, dy)
            out[b, h] = cur

    return out


def warp_mask_with_flow(mask, flow):
    if cv2 is None:
        return mask.astype(np.float32)

    h, w = mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 0.5).astype(np.float32)


def baseline_flow_warp(batch):
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
        for h in range(H):
            cur = warp_mask_with_flow(cur, flow)
            out[b, h] = cur

    return out


def run_baseline_eval(args):
    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.run_root) / "eval" / args.method / f"H{args.max_h}" / args.stress)
    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.csv"

    if args.skip_existing and rows_path.exists() and summary_path.exists():
        print(f"[skip] existing baseline eval: {summary_path}")
        return

    ds = DAVISObjectFutureDataset(
        dataset_root=args.dataset_root,
        split="val",
        year=args.year,
        prefix_len=args.prefix_len,
        max_h=args.max_h,
        image_size=args.image_size,
        stress=args.stress,
        random_stress=False,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_rows = []
    t0 = time.time()

    for step, batch in enumerate(dl):
        if args.method == "copy_last":
            pred = baseline_copy_last(batch)
        elif args.method == "linear_centroid":
            pred = baseline_linear_centroid(batch)
        elif args.method == "flow_warp":
            pred = baseline_flow_warp(batch)
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
            print(f"[baseline:{args.method}:{args.stress}:H{args.max_h}] step={step}/{len(dl)} rows={len(all_rows)}")

    df = pd.DataFrame(all_rows)
    summary = summarize_rows(df)

    df.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[done baseline] rows={rows_path}")
    print(f"[done baseline] summary={summary_path}")
    print(f"[time] {time.time() - t0:.1f}s")


# ------------------------------------------------------------
# Train and model evaluation
# ------------------------------------------------------------

def batch_to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def train_model(args):
    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.run_root) / "models" / args.method / f"H{args.max_h}")
    ckpt_last = out_dir / "last.pt"
    ckpt_best = out_dir / "best.pt"
    history_path = out_dir / "history.csv"
    config_path = out_dir / "config.json"

    if args.skip_existing and ckpt_best.exists():
        print(f"[skip] existing checkpoint: {ckpt_best}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] method={args.method} device={device} out={out_dir}")

    random_stress = bool(args.stress_train)
    train_ds = DAVISObjectFutureDataset(
        dataset_root=args.dataset_root,
        split="train",
        year=args.year,
        prefix_len=args.prefix_len,
        max_h=args.max_h,
        image_size=args.image_size,
        stress="clean",
        random_stress=random_stress,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_ds = DAVISObjectFutureDataset(
        dataset_root=args.dataset_root,
        split="val",
        year=args.year,
        prefix_len=args.prefix_len,
        max_h=args.max_h,
        image_size=args.image_size,
        stress="clean",
        random_stress=False,
        max_samples=min(args.max_val_samples, 512) if args.max_val_samples else 512,
        seed=args.seed,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args.method, image_size=args.image_size, max_h=args.max_h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_val = float("inf")
    history = []

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(batch["x_seq"], batch["geom"], sample=False)
                loss = bce_dice_loss(logits, batch["target"])
                if args.temporal_weight > 0:
                    loss = loss + args.temporal_weight * temporal_motion_loss(logits)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"[train:{args.method}:H{args.max_h}] "
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_dl)} "
                    f"loss={np.mean(train_losses[-20:]):.4f}"
                )

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                batch = batch_to_device(batch, device)
                logits = model(batch["x_seq"], batch["geom"], sample=False)
                loss = bce_dice_loss(logits, batch["target"])
                if args.temporal_weight > 0:
                    loss = loss + args.temporal_weight * temporal_motion_loss(logits)
                val_losses.append(float(loss.detach().cpu()))

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        row = {
            "epoch": epoch,
            "train_loss": tr,
            "val_loss": va,
            "seconds": time.time() - t0,
            "method": args.method,
            "max_h": args.max_h,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        ckpt = {
            "model": model.state_dict(),
            "method": args.method,
            "image_size": args.image_size,
            "max_h": args.max_h,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_last)

        if va < best_val:
            best_val = va
            torch.save(ckpt, ckpt_best)

        print(
            f"[epoch done:{args.method}:H{args.max_h}] "
            f"epoch={epoch} train={tr:.4f} val={va:.4f} best={best_val:.4f}"
        )

    print(f"[done train] best={ckpt_best}")


def eval_model(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt_path = Path(args.run_root) / "models" / args.method / f"H{args.max_h}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ensure_dir(Path(args.run_root) / "eval" / args.method / f"H{args.max_h}" / args.stress)
    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.csv"

    if args.skip_existing and rows_path.exists() and summary_path.exists():
        print(f"[skip] existing model eval: {summary_path}")
        return

    print(f"[eval] method={args.method} stress={args.stress} H={args.max_h} device={device}")

    ds = DAVISObjectFutureDataset(
        dataset_root=args.dataset_root,
        split="val",
        year=args.year,
        prefix_len=args.prefix_len,
        max_h=args.max_h,
        image_size=args.image_size,
        stress=args.stress,
        random_stress=False,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(args.method, image_size=args.image_size, max_h=args.max_h)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    stochastic_method = args.method in ["safeobj_full", "safeobj_no_geom", "safeobj_no_stress"]
    eval_k = args.eval_k if stochastic_method else 1

    all_rows = []
    t0 = time.time()

    with torch.no_grad():
        for step, batch_cpu in enumerate(dl):
            y_np = batch_cpu["target"].numpy()[:, :, 0]  # B,H,S,S
            batch = batch_to_device(batch_cpu, device)

            samples = []
            for k in range(eval_k):
                logits = model(batch["x_seq"], batch["geom"], sample=(eval_k > 1))
                probs = torch.sigmoid(logits).detach().cpu().numpy()  # B,H,1,S,S
                samples.append(probs)

            stack = np.stack(samples, axis=0)  # K,B,H,1,S,S
            mean_probs = stack.mean(axis=0)[:, :, 0]
            unc_map = stack.var(axis=0)[:, :, 0]
            unc_seq = unc_map.mean(axis=(-1, -2))  # B,H

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
                print(f"[eval:{args.method}:{args.stress}:H{args.max_h}] step={step}/{len(dl)} rows={len(all_rows)}")

    df = pd.DataFrame(all_rows)
    summary = summarize_rows(df)

    df.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[done eval] rows={rows_path}")
    print(f"[done eval] summary={summary_path}")
    print(f"[time] {time.time() - t0:.1f}s")


# ------------------------------------------------------------
# Manifest and aggregate
# ------------------------------------------------------------

def make_manifest(args):
    davis_root = locate_davis_root(args.dataset_root)

    out_dir = ensure_dir(Path(args.run_root))
    manifest_path = out_dir / "manifest_summary.json"

    info = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "davis_root": str(davis_root),
        "year": args.year,
        "prefix_len": args.prefix_len,
        "image_size": args.image_size,
        "max_h": args.max_h,
        "splits": {},
    }

    for split in ["train", "val"]:
        ds = DAVISObjectFutureDataset(
            dataset_root=args.dataset_root,
            split=split,
            year=args.year,
            prefix_len=args.prefix_len,
            max_h=args.max_h,
            image_size=args.image_size,
            stress="clean",
            random_stress=False,
            max_samples=0,
            seed=args.seed,
        )
        seqs = read_split(davis_root, args.year, split)
        info["splits"][split] = {
            "n_sequences": len(seqs),
            "n_samples": len(ds),
            "sequences": seqs,
        }
        print(f"[manifest] split={split} sequences={len(seqs)} samples={len(ds)}")

    with open(manifest_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[manifest] saved: {manifest_path}")


def aggregate(args):
    run_root = Path(args.run_root)
    summaries = sorted((run_root / "eval").rglob("summary.csv"))
    rows = sorted((run_root / "eval").rglob("rows.csv"))

    out_dir = ensure_dir(run_root / "RESULTS")
    if summaries:
        df_sum = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
        df_sum.to_csv(out_dir / "all_summaries.csv", index=False)
        print(f"[aggregate] summaries: {out_dir / 'all_summaries.csv'} rows={len(df_sum)}")

        # Main paper-ready tables
        df_all = df_sum[df_sum["h"].astype(str) == "all"].copy()
        if len(df_all):
            cols = [
                "method", "stress", "max_h",
                "iou", "boundary_f", "jf",
                "ops", "vanish", "hallucination", "split", "mce",
                "uncertainty", "uec", "ece_iou50", "n_samples"
            ]
            cols = [c for c in cols if c in df_all.columns]
            df_all[cols].to_csv(out_dir / "table_main_reliability.csv", index=False)
            print(f"[aggregate] table: {out_dir / 'table_main_reliability.csv'}")

    if rows:
        df_rows = pd.concat([pd.read_csv(p) for p in rows], ignore_index=True)
        df_rows.to_csv(out_dir / "all_rows.csv", index=False)
        print(f"[aggregate] rows: {out_dir / 'all_rows.csv'} rows={len(df_rows)}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

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
    p.add_argument("--run_root", type=str, default=str(Path.home() / "SafeObjWorld" / "runs_musthave"))

    p.add_argument("--year", type=str, default="2017")
    p.add_argument("--method", type=str, default="safeobj_full")
    p.add_argument("--stress", type=str, default="clean")

    p.add_argument("--prefix_len", type=int, default=3)
    p.add_argument("--max_h", type=int, default=5)
    p.add_argument("--image_size", type=int, default=192)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temporal_weight", type=float, default=0.05)

    p.add_argument("--stress_train", type=int, default=1)
    p.add_argument("--eval_k", type=int, default=5)

    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--log_every", type=int, default=25)

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 100)
    print("SafeObjWorld runner")
    print(json.dumps(vars(args), indent=2))
    print("=" * 100)

    if args.mode == "make_manifest":
        make_manifest(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "eval_model":
        eval_model(args)
    elif args.mode == "eval_baseline":
        run_baseline_eval(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
