import argparse, os, json
import pandas as pd
import torch, numpy as np
from tqdm import tqdm
from glob import glob

from src.embeddings.clip_backend import CLIPBackend
from src.presence.compute_presence import compute_alpha, inter_frame_divergence, boundary_prob, segment_weighted_presence

def list_frames(frames_root):
    frames = sorted(glob(os.path.join(frames_root, "*.jpg")) + glob(os.path.join(frames_root, "*.png")))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_root}")
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: frame_path,label (0/1)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.6)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--dry_caption", action="store_true", help="Skip Qwen; this pipeline uses CLIP only"); ap.add_argument("--qwen_json", default=None, help="Optional JSON from run_qwen_atoms.py"); ap.add_argument("--lambda_confirm", type=float, default=0.25); ap.add_argument("--lambda_resemble", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    frame_paths = df['frame_path'].tolist()
    labels = torch.tensor(df['label'].values, dtype=torch.float32)

    clip = CLIPBackend()

    # Optional Qwen bias
    qwen_bias = None
    if args.qwen_json and os.path.exists(args.qwen_json):
        from src.atoms.ontology import ALL_ATOMS
        atom_to_idx = {a:i for i,a in enumerate(sorted(ALL_ATOMS))}
        import json
        data = json.load(open(args.qwen_json))
        # build T x C bias matrix in [0,1]
        qbias = torch.zeros(len(frame_paths), len(atom_to_idx))
        for ti, fp in enumerate(frame_paths):
            rec = data.get(fp, {})
            for a in rec.get('confirm', []):
                if a in atom_to_idx:
                    qbias[ti, atom_to_idx[a]] = max(qbias[ti, atom_to_idx[a]], 1.0)
            for a in rec.get('resemble', []):
                if a in atom_to_idx and qbias[ti, atom_to_idx[a]] < 1.0:
                    qbias[ti, atom_to_idx[a]] = max(qbias[ti, atom_to_idx[a]], 0.5)
        qwen_bias = qbias

    with torch.no_grad():
        img_emb = clip.encode_image_paths(frame_paths)  # T x D
        p = clip.presence_scores(img_emb)               # T x C
        # Ensure everything is on the same device
        if qwen_bias is not None:
            qwen_bias = qwen_bias.to(p.device)
        # If Qwen atoms provided, softly bias presence toward 1 (confirm) / slight (resemble)
        if qwen_bias is not None and qwen_bias.shape == p.shape:
            # map bias->delta via convex blend: p' = p + λ*(bias*(1-p)) with separate λ
            confirm_mask = (qwen_bias >= 0.9).to(p.dtype) # .float()
            resemble_mask = ((qwen_bias > 0.0) & (qwen_bias < 0.9)).to(p.dtype) # .float()

            # λ are Python floats; cast to tensor on the same device for safety
            lam_c = torch.as_tensor(args.lambda_confirm, dtype=p.dtype, device=p.device)
            lam_r = torch.as_tensor(args.lambda_resemble, dtype=p.dtype, device=p.device)

            p = p + lam_c * confirm_mask * (1.0 - p) # args.lambda_confirm * confirm_mask * (1.0 - p)
            p = p + lam_r * resemble_mask * (1.0 - p) # args.lambda_resemble * resemble_mask * (1.0 - p)

        alpha = compute_alpha(p)                        # T x C
        d = inter_frame_divergence(alpha)               # T
        b = boundary_prob(d, args.delta, args.kappa)    # T
        pbar = segment_weighted_presence(p, b, args.window)  # T x C

    torch.save(p, os.path.join(args.out_dir, "presence.pt"))
    torch.save(b, os.path.join(args.out_dir, "boundary.pt"))
    torch.save(pbar, os.path.join(args.out_dir, "seg_presence.pt"))
    torch.save(labels, os.path.join(args.out_dir, "labels.pt"))

    # Save a small package for the trainer
    torch.save({"pbar": pbar, "labels": labels}, os.path.join(args.out_dir, "trainpack.pt"))
    print(f"Saved tensors in {args.out_dir}")

if __name__ == "__main__":
    main()
