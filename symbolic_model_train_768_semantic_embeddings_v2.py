#!/usr/bin/env python3
"""train_768_semantic_embeddings.py

This is a **drop-in replacement** for the previous flow-based discriminator
(normalizing flow) used on your precomputed 8-frame segment embeddings.

Instead of training a normalizing flow, this version trains a **symbolic
binary anomaly model** driven by X-CLIP cue prompts:

  - Model: CueBinaryPromptSymbolicNet (learnable per-event prompt + per-cue
    calibration; X-CLIP frozen)
  - Input: precomputed segment *video* embeddings (512-D) stored in your
    per-video PKLs under `captions_embedding` (only first 512 dims are used)
  - Output: p(abnormal) in [0,1]

It keeps your existing embedding I/O conventions:
  - train embeddings:
      train_embedding
      train_embedding_normal_segments_of_abnormal_videos
  - test embeddings:
      test_embedding

Pseudo augmentation
-------------------
If you have offline pseudo-aug files produced by `prepare_pseudo_aug_embeddings.py`,
this script can load them and add them to training.

Dependencies
------------
This script expects the folder from `symbolic_VLM_cooperate_v4.zip` to be
available on PYTHONPATH (or colocated near this file). We import:
  - xclip_cue_bank.build_cue_text_bank
  - xclip_wrapper.XCLIPEmbedder
  - symbolic_model.CueBinaryPromptSymbolicNet
  - gt_utils.parse_gt_file / match_gt_key_for_video

If imports fail, the script prints an actionable error explaining how to fix it.

Usage
-----
You can use it in two ways:

A) As a library (BGAD-style):
    from train_768_semantic_embeddings import train
    train(args)

B) As a standalone CLI:
    python train_768_semantic_embeddings.py train \
        --ckpt_out symbolic_binary_prompt.pt \
        --fp16 --batch_size 4096 --epochs 5 --lr 3e-4 \
        --use_offline_pseudo_aug

    python train_768_semantic_embeddings.py eval \
        --ckpt symbolic_binary_prompt.pt

Notes
-----
- We do **not** backprop through X-CLIP; we only use X-CLIP to encode cue texts
  once and then learn small prompt/calibration parameters.
- We evaluate on test embeddings by labeling a segment using the **mid-frame**
  index stored in `image` ("Frame_{mid}Crop_*") against GT intervals.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # pragma: no cover
    raise ImportError("This script requires PyTorch.") from e

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception as e:  # pragma: no cover
    raise ImportError("This script requires scikit-learn.") from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore


# -------------------------
# Constants / helpers
# -------------------------

MID_RE = re.compile(r"Frame_(\d+)")

CODE2LABEL = {
    "A": "normal",
    "B1": "fighting",
    "B2": "shooting",
    "B4": "riot",
    "B5": "abuse",
    "B6": "car accident",
    "G": "explosion",
}


def _tqdm(xs, **kwargs):
    if tqdm is None:
        return xs
    return tqdm(xs, **kwargs)


def extract_valid_512(vec: np.ndarray) -> np.ndarray:
    """Return the valid first 512 dims from a stored embedding.

    Your stored `captions_embedding` is often shape (1,768) where only the first
    512 dims are meaningful.
    """
    if vec is None:
        raise ValueError("captions_embedding is None")
    v = np.asarray(vec)
    if v.ndim != 1:
        v = v.reshape(-1)
    if v.shape[0] < 512:
        raise ValueError(f"Embedding dim {v.shape[0]} < 512")
    return v[:512].astype(np.float32, copy=False)


def parse_mid_frame(image_field: str) -> Optional[int]:
    if not isinstance(image_field, str):
        return None
    m = MID_RE.search(image_field)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _strip_suffixes(name: str) -> str:
    # Remove repeated suffixes like .pkl, .mp4, .avi
    s = str(name)
    for _ in range(3):
        base, ext = os.path.splitext(s)
        if ext.lower() in {".pkl", ".mp4", ".avi", ".mkv"}:
            s = base
        else:
            break
    return s


def extract_label_codes(video_id_or_filename: str) -> List[str]:
    """Robustly parse XDViolence label codes from a video id or embedding pkl name.

    Handles common variants:
      - ..._A.mp4
      - ..._B4-0-0.mp4
      - ..._label_B4-0-0.mp4
      - embedding PKLs that end with .pkl

    Returns:
        list of codes like ["A"] or ["B4","0","0"].
    """
    s = _strip_suffixes(video_id_or_filename)
    tail = s.split("_")[-1]

    # Remove leading "label_" if present
    tail = tail.replace("label_", "")

    parts = tail.split("-")

    # Clean each part (some pipelines leave stray suffix fragments)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Remove any accidental extension remnants
        p = _strip_suffixes(p)
        out.append(p)
    return out


def is_normal_video(video_id_or_filename: str) -> bool:
    codes = extract_label_codes(video_id_or_filename)
    # Normal videos usually have only code A (and optionally zeros)
    nonzero = [c for c in codes if c not in {"0", ""}]
    return (len(nonzero) == 1) and (nonzero[0] == "A")


def safe_video_label(video_id_or_filename: str) -> int:
    """Binary label for a video (0 normal, 1 abnormal) from filename codes."""
    return 0 if is_normal_video(video_id_or_filename) else 1


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return (x / n).astype(np.float32, copy=False)


# -------------------------
# Symbolic model imports
# -------------------------


def _maybe_add_symbolic_repo_to_path(symbolic_root: Optional[str] = None) -> None:
    """Try hard to make `symbolic_VLM_cooperate_v4` importable."""
    candidates: List[str] = []

    if symbolic_root:
        candidates.append(symbolic_root)

    env = os.environ.get("SYMBOLIC_VLM_COOPERATE_V4_ROOT")
    if env:
        candidates.append(env)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(here, "symbolic_VLM_cooperate_v4"),
            os.path.join(here, "..", "symbolic_VLM_cooperate_v4"),
            os.path.join(here, "..", "..", "symbolic_VLM_cooperate_v4"),
        ]
    )

    for p in candidates:
        if not p:
            continue
        p2 = os.path.abspath(p)
        if os.path.isdir(p2) and (p2 not in sys.path):
            sys.path.insert(0, p2)


def _import_symbolic_modules(symbolic_root: Optional[str] = None):
    _maybe_add_symbolic_repo_to_path(symbolic_root)

    try:
        from xclip_cue_bank import build_cue_text_bank
        from xclip_wrapper import XCLIPConfig, XCLIPEmbedder
        from symbolic_model import CueBinaryPromptSymbolicNet
        from gt_utils import match_gt_key_for_video, parse_gt_file

        return build_cue_text_bank, XCLIPConfig, XCLIPEmbedder, CueBinaryPromptSymbolicNet, parse_gt_file, match_gt_key_for_video
    except Exception as e:
        msg = (
            "[ERROR] Failed to import symbolic X-CLIP modules.\n\n"
            "Expected the extracted `symbolic_VLM_cooperate_v4` folder (from your zip) to be importable.\n"
            "Fix options:\n"
            "  1) Put `symbolic_VLM_cooperate_v4/` next to this file, OR\n"
            "  2) Export SYMBOLIC_VLM_COOPERATE_V4_ROOT=/path/to/symbolic_VLM_cooperate_v4, OR\n"
            "  3) Add that folder to PYTHONPATH.\n\n"
            f"Original import error: {type(e).__name__}: {e}\n"
        )
        raise ImportError(msg) from e


# -------------------------
# Dataset
# -------------------------


class EmbeddingDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[np.ndarray, int]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        # x is np.ndarray (512,), y is int
        return torch.from_numpy(x), torch.tensor(float(y), dtype=torch.float32)


# -------------------------
# Load embeddings
# -------------------------


def load_embedding_pkl(path: str) -> List[np.ndarray]:
    """Load one per-video embedding PKL and return list of 512-D vectors."""
    segs = joblib.load(path)
    if not isinstance(segs, list):
        return []

    out: List[np.ndarray] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        emb = seg.get("captions_embedding", None)
        if emb is None:
            continue
        try:
            v = np.asarray(emb)[0]
        except Exception:
            continue
        try:
            v512 = extract_valid_512(v)
        except Exception:
            continue
        out.append(v512)
    return out


def load_train_samples(
    src_dir_train: str,
    src_dir_train_normal_segments_of_abnormal_videos: str,
    *,
    use_offline_pseudo_aug: bool,
    pseudo_aug_normal_pkl: str,
    pseudo_aug_abnormal_pkl: str,
    use_pseudo_negative_aug: bool,
    max_train_samples: int = 0,
    seed: int = 0,
) -> List[Tuple[np.ndarray, int]]:
    """Build training samples [(vec512,label)]."""
    rng = np.random.default_rng(int(seed))

    samples: List[Tuple[np.ndarray, int]] = []

    # # 1) Train embeddings from all videos
    # if os.path.isdir(src_dir_train):
    #     for fn in sorted(os.listdir(src_dir_train)):
    #         if not fn.endswith(".pkl"):
    #             continue
    #         lab = safe_video_label(fn)
    #         p = os.path.join(src_dir_train, fn)
    #         try:
    #             vecs = load_embedding_pkl(p)
    #         except Exception:
    #             continue
    #         for v in vecs:
    #             samples.append((v, lab))
    #
    # # 2) Extra normal segments mined from abnormal videos
    # if os.path.isdir(src_dir_train_normal_segments_of_abnormal_videos):
    #     for fn in sorted(os.listdir(src_dir_train_normal_segments_of_abnormal_videos)):
    #         if not fn.endswith(".pkl"):
    #             continue
    #         # Keep original guard: skip normal-video files if they exist here.
    #         if "label_A" in fn or fn.endswith("_A.pkl"):
    #             continue
    #
    #         p = os.path.join(src_dir_train_normal_segments_of_abnormal_videos, fn)
    #         try:
    #             vecs = load_embedding_pkl(p)
    #         except Exception:
    #             continue
    #         for v in vecs:
    #             samples.append((v, 0))

    # 3) Offline pseudo augmentation (prepared by prepare_pseudo_aug_embeddings.py)
    if use_offline_pseudo_aug:
        if os.path.exists(pseudo_aug_abnormal_pkl):
            try:
                abnormal_aug = joblib.load(pseudo_aug_abnormal_pkl)
            except Exception:
                abnormal_aug = []
        else:
            abnormal_aug = []

        if os.path.exists(pseudo_aug_normal_pkl):
            try:
                normal_aug = joblib.load(pseudo_aug_normal_pkl)
            except Exception:
                normal_aug = []
        else:
            normal_aug = []

        # Files are lists like: [vec512, [label]]
        for item in abnormal_aug:
            try:
                v = np.asarray(item[0], dtype=np.float32)
                samples.append((v[:512].astype(np.float32, copy=False), 1))
            except Exception:
                continue

        if use_pseudo_negative_aug:
            for item in normal_aug:
                try:
                    v = np.asarray(item[0], dtype=np.float32)
                    samples.append((v[:512].astype(np.float32, copy=False), 0))
                except Exception:
                    continue

        print(
            f"[INFO] Offline pseudo-aug loaded: +{len(abnormal_aug)} pos, +{len(normal_aug) if use_pseudo_negative_aug else 0} neg"
        )

    # 4) Shuffle and optionally cap
    rng.shuffle(samples)
    if max_train_samples and max_train_samples > 0 and len(samples) > int(max_train_samples):
        samples = samples[: int(max_train_samples)]

    return samples


def _interval_contains_mid(intervals_inclusive: Sequence[Tuple[int, int]], mid: int, *, end_exclusive: bool) -> bool:
    m = int(mid)
    for s, e in intervals_inclusive:
        s2 = int(s)
        e2 = int(e)
        if end_exclusive:
            if s2 <= m < e2:
                return True
        else:
            if s2 <= m <= e2:
                return True
    return False


def load_test_samples(
    src_dir_test: str,
    gt_txt: str,
    *,
    symbolic_root: Optional[str] = None,
    gt_end_exclusive: bool,
    max_test_samples: int = 0,
) -> Tuple[
    List[Tuple[np.ndarray, int]],
    List[str],
    List[str],
    int,
    int,
]:
    """Build test samples and also return record lists for compatibility.

    Returns:
        all_samples: [(vec512,label)] concatenated as normal first then abnormal
        record_normal_ids: list[str]
        record_abnormal_ids: list[str]
        n_normal: number of normal samples (prefix length)
        n_abnormal: number of abnormal samples
    """
    build_cue_text_bank, XCLIPConfig, XCLIPEmbedder, CueBinaryPromptSymbolicNet, parse_gt_file, match_gt_key_for_video = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    # The above assignment is just to keep linters quiet; we don't import symbolic
    # modules here. GT parsing comes from symbolic repo though, so we import it.
    (_bank, _cfg, _emb, _model_cls, parse_gt_file, match_gt_key_for_video) = _import_symbolic_modules(symbolic_root)

    gt_map = parse_gt_file(gt_txt)
    gt_keys = list(gt_map.keys())

    normal_samples: List[Tuple[np.ndarray, int]] = []
    abnormal_samples: List[Tuple[np.ndarray, int]] = []
    record_normal: List[str] = []
    record_abnormal: List[str] = []

    if not os.path.isdir(src_dir_test):
        return [], [], [], 0, 0

    for fn in sorted(os.listdir(src_dir_test)):
        if not fn.endswith(".pkl"):
            continue
        p = os.path.join(src_dir_test, fn)
        try:
            segs = joblib.load(p)
        except Exception:
            continue
        if not isinstance(segs, list) or len(segs) == 0:
            continue

        # Match GT key
        gt_key = match_gt_key_for_video(fn, gt_keys)
        intervals = gt_map.get(gt_key, []) if gt_key is not None else []

        for seg_idx, seg in enumerate(segs):
            if not isinstance(seg, dict):
                continue
            mid = parse_mid_frame(seg.get("image", ""))
            if mid is None:
                continue

            emb = seg.get("captions_embedding", None)
            if emb is None:
                continue
            try:
                v = np.asarray(emb)[0]
                vec512 = extract_valid_512(v)
            except Exception:
                continue

            lab = 1 if _interval_contains_mid(intervals, mid, end_exclusive=gt_end_exclusive) else 0

            if lab == 0:
                normal_samples.append((vec512, 0))
                record_normal.append(f"{fn}: {seg_idx}")
            else:
                abnormal_samples.append((vec512, 1))
                record_abnormal.append(f"{fn}: {seg_idx}")

            if max_test_samples and max_test_samples > 0:
                if (len(normal_samples) + len(abnormal_samples)) >= int(max_test_samples):
                    break
        if max_test_samples and max_test_samples > 0:
            if (len(normal_samples) + len(abnormal_samples)) >= int(max_test_samples):
                break

    all_samples = normal_samples + abnormal_samples
    return all_samples, record_normal, record_abnormal, len(normal_samples), len(abnormal_samples)


# -------------------------
# Model build / train / eval
# -------------------------


@dataclass
class SymbolicBuildResult:
    model: torch.nn.Module
    bank: object
    device: torch.device


def build_symbolic_prompt_model(
    *,
    symbolic_root: Optional[str],
    xclip_model_name: str,
    device_str: str,
    fp16_text: bool,
    batch_text: int,
    prompt_dim: int,
    base_text_embs: Optional[np.ndarray] = None,
) -> SymbolicBuildResult:
    build_cue_text_bank, XCLIPConfig, XCLIPEmbedder, CueBinaryPromptSymbolicNet, parse_gt_file, match_gt_key_for_video = (
        _import_symbolic_modules(symbolic_root)
    )

    device = torch.device(device_str if torch.cuda.is_available() and str(device_str).startswith("cuda") else "cpu")

    bank = build_cue_text_bank()

    # Encode cue text variants once using the X-CLIP text tower.
    #
    # If `base_text_embs` is provided (e.g. loaded from a checkpoint), we can
    # skip instantiating X-CLIP (and thus skip requiring `transformers`) which
    # makes eval-only runs lighter.
    if base_text_embs is None:
        cfg = XCLIPConfig(
            model_name=str(xclip_model_name),
            device=str(device),
            fp16=bool(fp16_text),
            batch_text=int(batch_text),
            batch_video=1,
        )
        embedder = XCLIPEmbedder(cfg)
        text_embs = embedder.encode_texts(bank.texts)  # (T,D)
    else:
        text_embs = np.asarray(base_text_embs, dtype=np.float32)

    if text_embs.ndim != 2:
        raise RuntimeError(f"Unexpected base_text_embs shape: {text_embs.shape}")

    model = CueBinaryPromptSymbolicNet(
        cue_keys=bank.cue_keys,
        cue_descs_by_event=bank.cue_descs_by_event,
        cue_to_text_indices=bank.cue_to_text_indices,
        text_event_idx=bank.text_event_idx,
        base_text_embs=text_embs,
        prompt_dim=int(prompt_dim),
    ).to(device)

    return SymbolicBuildResult(model=model, bank=bank, device=device)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))



def _make_cpu_generator(seed: int) -> torch.Generator:
    """Create a CPU RNG generator for DataLoader shuffling.

    This avoids the error:
        RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'
    which can happen when a CUDA generator is used implicitly.
    """
    try:
        g = torch.Generator(device="cpu")  # newer PyTorch supports device kwarg
    except TypeError:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def train_symbolic_model(
    model: torch.nn.Module,
    train_samples: Sequence[Tuple[np.ndarray, int]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    fp16_train: bool,
    num_workers: int,
) -> None:
    _set_seed(seed)

    ds = EmbeddingDataset(train_samples)
    g = _make_cpu_generator(seed)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        generator=g,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    use_amp = bool(fp16_train) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(int(epochs)):
        total_loss = 0.0
        total = 0

        it = _tqdm(loader, desc=f"[TRAIN] epoch {ep+1}/{int(epochs)}", unit="batch")
        for xb, yb in it:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # L2 normalize embeddings to be safe (X-CLIP uses cosine space)
            xb = F.normalize(xb, dim=1)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                p_ab, _ev, _cp = model(xb)
                p_ab = torch.clamp(p_ab, 1e-6, 1.0 - 1e-6)
                loss = torch.nn.functional.binary_cross_entropy(p_ab, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(xb.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            total += bs

            if tqdm is not None:
                it.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")

        avg = total_loss / max(1, total)
        print(f"[TRAIN] epoch {ep+1:03d}/{int(epochs)} | loss={avg:.6f} | n={total}")


@torch.no_grad()
def predict_symbolic(
    model: torch.nn.Module,
    samples: Sequence[Tuple[np.ndarray, int]],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = EmbeddingDataset(samples)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.eval()

    preds: List[float] = []
    gts: List[int] = []

    for xb, yb in _tqdm(loader, desc="[EVAL]", unit="batch"):
        xb = xb.to(device, non_blocking=True)
        xb = F.normalize(xb, dim=1)
        p_ab, _ev, _cp = model(xb)
        preds.extend(p_ab.detach().cpu().numpy().astype(np.float32).tolist())
        gts.extend(yb.detach().cpu().numpy().astype(np.int64).tolist())

    return np.asarray(preds, dtype=np.float32), np.asarray(gts, dtype=np.int64)



@torch.no_grad()
def _predict_scores_for_vecs(
    model: torch.nn.Module,
    vecs512: Sequence[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run the symbolic model on a list of 512-D vectors and return p(abnormal)."""
    if not vecs512:
        return np.zeros((0,), dtype=np.float32)

    model.eval()

    scores: List[float] = []
    bs = int(batch_size)
    for start in range(0, len(vecs512), bs):
        chunk = vecs512[start : start + bs]
        xb = torch.from_numpy(np.stack([np.asarray(v, dtype=np.float32)[:512] for v in chunk], axis=0))
        xb = xb.to(device, non_blocking=True)
        xb = F.normalize(xb, dim=1)
        p_ab, _ev, _cp = model(xb)
        scores.extend(p_ab.detach().cpu().numpy().astype(np.float32).tolist())
    return np.asarray(scores, dtype=np.float32)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    w = int(win)
    if w <= 1 or x.size == 0:
        return x
    k = np.ones((w,), dtype=np.float32) / float(w)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xp, k, mode="valid")
    return y.astype(np.float32, copy=False)


def _build_gt_curve(
    intervals: Sequence[Tuple[int, int]],
    *,
    length: int,
    end_exclusive: bool,
) -> np.ndarray:
    gt = np.zeros((int(length),), dtype=np.float32)
    if length <= 0:
        return gt
    for s, e in intervals:
        try:
            s2 = int(s)
            e2 = int(e)
        except Exception:
            continue
        if end_exclusive:
            a = max(0, min(length, s2))
            b = max(0, min(length, e2))
            if b > a:
                gt[a:b] = 1.0
        else:
            a = max(0, min(length - 1, s2))
            b = max(0, min(length - 1, e2))
            if b >= a:
                gt[a : b + 1] = 1.0
    return gt


def _segment_scores_to_frame_curve(
    mids: Sequence[int],
    seg_scores: Sequence[float],
    *,
    length: int,
    mode: str,
    seg_len: int,
    mid_offset: int,
) -> np.ndarray:
    """Convert segment scores (at mid-frames) into a per-frame curve."""
    L = int(length)
    if L <= 0:
        return np.zeros((0,), dtype=np.float32)

    mode = str(mode or "fill_max").lower()
    seg_len = int(seg_len)
    mid_offset = int(mid_offset)

    if mode == "mid_interp":
        curve = np.full((L,), np.nan, dtype=np.float32)
        for m, sc in zip(mids, seg_scores):
            try:
                mi = int(m)
                s = float(sc)
            except Exception:
                continue
            if 0 <= mi < L:
                if np.isnan(curve[mi]):
                    curve[mi] = s
                else:
                    curve[mi] = max(float(curve[mi]), s)
        idx = np.where(np.isfinite(curve))[0]
        if idx.size == 0:
            return np.zeros((L,), dtype=np.float32)
        curve = np.interp(np.arange(L), idx, curve[idx]).astype(np.float32)
        return curve

    if mode == "fill_mean":
        acc = np.zeros((L,), dtype=np.float32)
        cnt = np.zeros((L,), dtype=np.float32)
        for m, sc in zip(mids, seg_scores):
            try:
                mi = int(m)
                s = float(sc)
            except Exception:
                continue
            a = mi - mid_offset
            b = a + seg_len - 1
            a = max(0, a)
            b = min(L - 1, b)
            if b >= a:
                acc[a : b + 1] += s
                cnt[a : b + 1] += 1.0
        out = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
        return out.astype(np.float32, copy=False)

    # default: fill_max
    curve = np.zeros((L,), dtype=np.float32)
    for m, sc in zip(mids, seg_scores):
        try:
            mi = int(m)
            s = float(sc)
        except Exception:
            continue
        a = mi - mid_offset
        b = a + seg_len - 1
        a = max(0, a)
        b = min(L - 1, b)
        if b >= a:
            curve[a : b + 1] = np.maximum(curve[a : b + 1], s)
    return curve.astype(np.float32, copy=False)


def _try_get_video_frame_count(video_dir: str, video_stem: str) -> Optional[int]:
    """Best-effort frame count using cv2 (optional dependency)."""
    if not video_dir:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    exts = [".mp4", ".avi", ".mkv", ""]
    for ext in exts:
        cand = os.path.join(video_dir, video_stem + ext)
        if not os.path.exists(cand):
            continue
        cap = cv2.VideoCapture(cand)
        if not cap.isOpened():
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    return None


def plot_per_video_curves(
    model: torch.nn.Module,
    *,
    device: torch.device,
    src_dir_test: str,
    gt_txt: str,
    symbolic_root: Optional[str],
    gt_end_exclusive: bool,
    out_dir: str,
    plot_mode: str,
    plot_video_filter: str,
    plot_max_videos: int,
    plot_batch_size: int,
    emb_seg_len: int,
    emb_mid_offset: int,
    plot_smooth: int,
    video_dir: str,
) -> None:
    """Save per-video curves as PNG: predicted anomaly score vs frame index, plus GT curve."""
    os.makedirs(out_dir, exist_ok=True)

    (_bank, _cfg, _emb, _model_cls, parse_gt_file, match_gt_key_for_video) = _import_symbolic_modules(symbolic_root)
    gt_map = parse_gt_file(gt_txt)
    gt_keys = list(gt_map.keys())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    filt = (plot_video_filter or "").strip().lower()
    max_v = int(plot_max_videos)

    video_pkls = [fn for fn in sorted(os.listdir(src_dir_test)) if fn.endswith(".pkl")]
    num_done = 0

    for fn in video_pkls:
        if filt and (filt not in fn.lower()):
            continue

        p = os.path.join(src_dir_test, fn)
        try:
            segs = joblib.load(p)
        except Exception:
            continue
        if not isinstance(segs, list) or len(segs) == 0:
            continue

        gt_key = match_gt_key_for_video(fn, gt_keys)
        intervals = gt_map.get(gt_key, []) if gt_key is not None else []

        mids: List[int] = []
        vecs: List[np.ndarray] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            mid = parse_mid_frame(seg.get("image", ""))
            if mid is None:
                continue
            emb = seg.get("captions_embedding", None)
            if emb is None:
                continue
            try:
                v = np.asarray(emb)[0]
                v512 = extract_valid_512(v)
            except Exception:
                continue
            mids.append(int(mid))
            vecs.append(v512)

        if not vecs:
            continue

        seg_scores = _predict_scores_for_vecs(
            model,
            vecs,
            device=device,
            batch_size=int(plot_batch_size),
        )

        max_mid = int(max(mids)) if mids else 0
        max_seg_e = int(max([m - int(emb_mid_offset) + int(emb_seg_len) - 1 for m in mids])) if mids else max_mid
        max_gt_end_used = 0
        for s, e in intervals:
            try:
                s2 = int(s)
                e2 = int(e)
            except Exception:
                continue
            used = (e2 - 1) if gt_end_exclusive else e2
            max_gt_end_used = max(max_gt_end_used, used, s2)

        L = max(max_mid, max_seg_e, max_gt_end_used) + 1

        stem = os.path.splitext(fn)[0]
        stem2 = stem[:-4] if stem.lower().endswith(".mp4") else stem
        n_frames = _try_get_video_frame_count(video_dir, stem2)
        if n_frames is not None and n_frames > 0:
            L = int(n_frames)

        pred_curve = _segment_scores_to_frame_curve(
            mids,
            seg_scores.tolist(),
            length=L,
            mode=str(plot_mode),
            seg_len=int(emb_seg_len),
            mid_offset=int(emb_mid_offset),
        )
        if int(plot_smooth) and int(plot_smooth) > 1:
            pred_curve = _moving_average(pred_curve, int(plot_smooth))

        gt_curve = _build_gt_curve(intervals, length=L, end_exclusive=bool(gt_end_exclusive))

        x = np.arange(L, dtype=np.int32)
        plt.figure(figsize=(14, 4))
        plt.plot(x, pred_curve, label="pred(p_abnormal)")
        plt.step(x, gt_curve, where="post", label="gt", linewidth=1.5)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("frame index")
        plt.ylabel("anomaly score")
        plt.title(stem)
        plt.legend(loc="upper right")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{stem}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        num_done += 1
        if max_v > 0 and num_done >= max_v:
            break

    print(f"[INFO] Saved per-video curves to: {out_dir} (videos={num_done})")
def safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)

    # If only one class is present, roc_auc_score throws.
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    return auc, ap


def save_ckpt(path: str, model: torch.nn.Module, bank: object, meta: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "cue_keys": getattr(bank, "cue_keys", None),
        "event_order_6": getattr(sys.modules.get("symbolic_model", None), "EVENT_ORDER_6", None),
        "class_order": ["normal", "abnormal"],
        "use_learnable_prompt": bool(getattr(model, "expects_video_embeddings", False)),
        # Convenience fields for eval-only loading without re-encoding texts.
        "prompt_dim": int(getattr(model, "prompt_dim", 0) or 0),
        "text_dim": int(getattr(model, "text_dim", 0) or 0),
        "meta": dict(meta or {}),
    }
    torch.save(ckpt, path)


def load_ckpt(path: str) -> Dict[str, object]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Bad checkpoint: {path}")
    return obj


# -------------------------
# BGAD-style entrypoint
# -------------------------


def validate(*_args, **_kwargs):
    """Compatibility stub.

    The original BGAD engine expects an image-based `validate()` that evaluates
    a normalizing flow on backbone feature maps.

    This rewritten file trains a symbolic prompt model on **precomputed
    embeddings**, so image-based validation is not applicable.
    """

    print(
        "[WARN] validate() is not implemented for the symbolic prompt model. "
        "This script evaluates using the embedding test set inside train()."
    )
    return float("nan"), float("nan"), float("nan")


def train(args) -> Tuple[float, float, float]:
    """BGAD-style entrypoint.

    Returns:
        (auc, ap, dummy)
    """

    # Directories (keep your original defaults unless overridden)
    src_dir_train = getattr(
        args,
        "src_dir_train",
        "train_embedding",
    )
    src_dir_train_normal_segments_of_abnormal_videos = getattr(
        args,
        "src_dir_train_normal_segments_of_abnormal_videos",
        "train_embedding_normal_segments_of_abnormal_videos",
    )
    src_dir_test = getattr(
        args,
        "src_dir_test",
        "test_embedding",
    )

    # GT file for test
    gt_txt = getattr(
        args,
        "test_gt_txt",
        "annotations_standard.txt",
    )

    gt_end_exclusive = bool(getattr(args, "gt_end_exclusive", False))

    # Offline pseudo aug
    use_offline_pseudo_aug = bool(getattr(args, "use_offline_pseudo_aug", True))
    pseudo_aug_normal_pkl = getattr(
        args,
        "pseudo_aug_normal_pkl",
        "pseudo_aug_out/pseudo_aug_normal.pkl",
    )
    pseudo_aug_abnormal_pkl = getattr(
        args,
        "pseudo_aug_abnormal_pkl",
        "pseudo_aug_out/pseudo_aug_abnormal.pkl",
    )
    use_pseudo_negative_aug = bool(getattr(args, "use_pseudo_negative_aug", True))

    # Build train samples
    train_samples = load_train_samples(
        src_dir_train,
        src_dir_train_normal_segments_of_abnormal_videos,
        use_offline_pseudo_aug=use_offline_pseudo_aug,
        pseudo_aug_normal_pkl=pseudo_aug_normal_pkl,
        pseudo_aug_abnormal_pkl=pseudo_aug_abnormal_pkl,
        use_pseudo_negative_aug=use_pseudo_negative_aug,
        max_train_samples=int(getattr(args, "max_train_samples", 0) or 0),
        seed=int(getattr(args, "seed", 0) or 0),
    )
    print(f"[INFO] Train samples: {len(train_samples)}")

    # Symbolic repo root (needed for GT parsing + model)
    symbolic_root = getattr(args, "symbolic_root", None)

    # Build test samples
    test_samples, record_normal, record_abnormal, n_normal, n_abnormal = load_test_samples(
        src_dir_test,
        gt_txt,
        symbolic_root=symbolic_root,
        gt_end_exclusive=gt_end_exclusive,
        max_test_samples=int(getattr(args, "max_test_samples", 0) or 0),
    )
    print(f"[INFO] Test samples: {len(test_samples)} (normal={n_normal}, abnormal={n_abnormal})")

    # Build symbolic model
    xclip_model_name = getattr(args, "xclip_model_name", "frames_encoder_based_on_clip")
    device_str = getattr(args, "device", "cuda")
    fp16 = bool(getattr(args, "fp16", True))
    batch_text = int(getattr(args, "batch_text", 256))

    # If a checkpoint is provided, we can reuse its stored base text embeddings
    # to avoid re-encoding cue texts with X-CLIP (faster + no transformers needed
    # for eval-only runs).
    ckpt_path = getattr(args, "checkpoint", None) or getattr(args, "ckpt", None)
    ckpt = None
    base_text_embs = None
    prompt_dim = int(getattr(args, "prompt_dim", 512))
    if ckpt_path:
        ckpt = load_ckpt(str(ckpt_path))
        try:
            pd = int(ckpt.get("prompt_dim", 0) or 0)
            if pd > 0:
                prompt_dim = pd
        except Exception:
            pass

        try:
            bt = (ckpt.get("state_dict", {}) or {}).get("base_text_embs", None)
            if isinstance(bt, torch.Tensor):
                base_text_embs = bt.detach().cpu().numpy()
        except Exception:
            base_text_embs = None

    build = build_symbolic_prompt_model(
        symbolic_root=symbolic_root,
        xclip_model_name=xclip_model_name,
        device_str=device_str,
        fp16_text=fp16,
        batch_text=batch_text,
        prompt_dim=prompt_dim,
        base_text_embs=base_text_embs,
    )
    model = build.model
    device = build.device

    # Load checkpoint weights after constructing the model
    if ckpt_path and ckpt is not None:
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # Train
    epochs = int(getattr(args, "epochs", 0) or getattr(args, "meta_epochs", 5) or 5)
    lr = float(getattr(args, "lr", 3e-4))
    batch_size = int(getattr(args, "batch_size", 4096))
    seed = int(getattr(args, "seed", 0))
    fp16_train = bool(getattr(args, "fp16_train", False))
    num_workers = int(getattr(args, "num_workers", 4))

    eval_only = bool(getattr(args, "eval_only", False))

    if not eval_only:
        train_symbolic_model(
            model,
            train_samples,
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            fp16_train=fp16_train,
            num_workers=num_workers,
        )

    # Eval
    preds, gts = predict_symbolic(model, test_samples, device=device, batch_size=batch_size, num_workers=num_workers)
    auc, ap = safe_auc_ap(gts, preds)
    print(f"[RESULT] Segment-AUC: {auc:.6f} | AP: {ap:.6f}")

    # Compatibility dumps (same filenames as your flow version)
    joblib.dump(record_normal, "normal_frames_id_lib.pkl")
    joblib.dump(record_abnormal, "abnormal_frames_id_lib.pkl")

    joblib.dump(preds[:n_normal].tolist(), "normal_frames_pred_lib.pkl")
    joblib.dump(preds[n_normal:].tolist(), "abnormal_frames_pred_lib.pkl")

    # Save checkpoint
    ckpt_out = getattr(args, "ckpt_out", None) or getattr(args, "output_ckpt", None)
    if ckpt_out:
        save_ckpt(
            str(ckpt_out),
            model,
            build.bank,
            meta={
                "xclip_model_name": str(xclip_model_name),
                "prompt_dim": int(prompt_dim),
                "gt_txt": str(gt_txt),
                "gt_end_exclusive": bool(gt_end_exclusive),
                "epochs": int(epochs),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "seed": int(seed),
            },
        )
        print(f"[INFO] Saved checkpoint: {ckpt_out}")


    # Optional: save per-video anomaly score curves
    if bool(getattr(args, "plot_curves", False)):
        plot_per_video_curves(
            model,
            device=device,
            src_dir_test=src_dir_test,
            gt_txt=gt_txt,
            symbolic_root=symbolic_root,
            gt_end_exclusive=gt_end_exclusive,
            out_dir=str(getattr(args, "plot_out_dir", "curve_plots")),
            plot_mode=str(getattr(args, "plot_mode", "fill_max")),
            plot_video_filter=str(getattr(args, "plot_video_filter", "")),
            plot_max_videos=int(getattr(args, "plot_max_videos", 0) or 0),
            plot_batch_size=int(getattr(args, "plot_batch_size", batch_size) or batch_size),
            emb_seg_len=int(getattr(args, "emb_seg_len", 8) or 8),
            emb_mid_offset=int(getattr(args, "emb_mid_offset", 4) or 4),
            plot_smooth=int(getattr(args, "plot_smooth", 0) or 0),
            video_dir=str(getattr(args, "video_dir", "")),
        )

    return float(auc), float(ap), float("nan")


# -------------------------
# Standalone CLI
# -------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--device", type=str, default="cuda:1")
        sp.add_argument("--fp16", action="store_true", help="Use fp16 for X-CLIP text encoding (CUDA only).")
        sp.add_argument("--batch_text", type=int, default=256)
        sp.add_argument("--prompt_dim", type=int, default=512)
        sp.add_argument("--xclip_model_name", type=str, default="frames_encoder_based_on_clip")
        sp.add_argument(
            "--symbolic_root",
            type=str,
            default=None,
            help="Path to extracted symbolic_VLM_cooperate_v4 folder (optional).",
        )

        sp.add_argument(
            "--src_dir_train",
            type=str,
            default="train_embedding",
        )
        sp.add_argument(
            "--src_dir_train_normal_segments_of_abnormal_videos",
            type=str,
            default="train_embedding_normal_segments_of_abnormal_videos",
        )
        sp.add_argument(
            "--src_dir_test",
            type=str,
            default="test_embedding",
        )

        sp.add_argument(
            "--test_gt_txt",
            type=str,
            default="annotations_standard.txt",
        )
        sp.add_argument(
            "--gt_end_exclusive",
            action="store_true",
            help="Treat GT intervals as [start,end) instead of inclusive [start,end].",
        )

        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max_train_samples", type=int, default=0)
        sp.add_argument("--max_test_samples", type=int, default=0)
        sp.add_argument("--num_workers", type=int, default=0)

        # --- Visualization: per-video anomaly score curves (pred vs GT) ---
        sp.add_argument(
            "--plot_curves",
            action="store_true",
            help="Save per-video anomaly score curves (predicted score vs per-frame GT) as PNGs.",
        )
        sp.add_argument(
            "--plot_out_dir",
            type=str,
            default="curves_test",
            help="Output directory for per-video curve PNGs.",
        )
        sp.add_argument(
            "--plot_mode",
            type=str,
            default="fill_max",
            choices=["fill_max", "fill_mean", "mid_interp"],
            help="How to convert segment scores to per-frame scores.",
        )
        sp.add_argument(
            "--plot_video_filter",
            type=str,
            default="",
            help="Only plot videos whose embedding filename contains this substring (optional).",
        )
        sp.add_argument(
            "--plot_max_videos",
            type=int,
            default=0,
            help="Max number of videos to plot (0 = all).",
        )
        sp.add_argument(
            "--plot_batch_size",
            type=int,
            default=2048,
            help="Batch size used when running the model for plotting.",
        )
        sp.add_argument(
            "--video_dir",
            type=str,
            default="/media/yons/PortableSSD/Dataset/XDViolence/videos/videos",
            help="Optional raw video directory to get exact frame count via cv2 (if available).",
        )
        sp.add_argument(
            "--emb_seg_len",
            type=int,
            default=8,
            help="Segment length (in frames) represented by each embedding.",
        )
        sp.add_argument(
            "--emb_mid_offset",
            type=int,
            default=4,
            help="Segment start = mid - emb_mid_offset.",
        )
        sp.add_argument(
            "--plot_smooth",
            type=int,
            default=0,
            help="Optional moving-average window for smoothing predicted scores (0 disables).",
        )


        sp.add_argument("--use_offline_pseudo_aug", action="store_true")
        sp.add_argument("--use_pseudo_negative_aug", action="store_true")
        sp.add_argument(
            "--pseudo_aug_normal_pkl",
            type=str,
            default="pseudo_aug_out/pseudo_aug_normal.pkl",
        )
        sp.add_argument(
            "--pseudo_aug_abnormal_pkl",
            type=str,
            default="pseudo_aug_out/pseudo_aug_abnormal.pkl",
        )

    # train
    p_train = sub.add_parser("train", help="Train and evaluate the symbolic prompt model")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--batch_size", type=int, default=40960)
    p_train.add_argument(
        "--fp16_train",
        action="store_true",
        help="Use amp during prompt/calibration training (CUDA only).",
    )
    p_train.add_argument("--ckpt_out", type=str, default="symbolic_binary_prompt.pt")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate from a checkpoint")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--batch_size", type=int, default=4096)

    return p


def main() -> None:
    args = _build_argparser().parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        args.eval_only = True
        args.checkpoint = args.ckpt
        # We still pass ckpt_out=None to avoid overwriting by default.
        args.ckpt_out = None
        train(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
