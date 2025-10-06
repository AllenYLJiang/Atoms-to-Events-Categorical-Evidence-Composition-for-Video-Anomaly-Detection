#!/usr/bin/env python3
import os, argparse, json
import torch, numpy as np, pandas as pd
from typing import List, Tuple
from src.train.role_learning import RoleLearner, loss_video_level
from src.atoms.ontology import EVENTS, EVENT2ATOMS, ALL_ATOMS

def load_run(run_dir: str, device: str):
    pack = torch.load(os.path.join(run_dir, "trainpack.pt"), map_location=device)
    pbar = pack["pbar"].to(device)   # T x C
    return pbar

def load_Y(video_labels_csv: str, device: str):
    df = pd.read_csv(video_labels_csv)
    assert list(df.columns[1:]) == EVENTS, f"Columns must be: ['video_id'] + {EVENTS}"
    y = torch.tensor(df.iloc[0, 1:].values.astype(np.float32), device=device)  # E,
    return y.unsqueeze(0)  # 1 x E

def build_manifest(manifest_csv: str) -> List[Tuple[str, str]]:
    """CSV with columns: run_dir,labels_csv"""
    df = pd.read_csv(manifest_csv)
    req = {"run_dir", "labels_csv"}
    assert req.issubset(df.columns), f"manifest must have columns {req}"
    return [(r, l) for r, l in zip(df["run_dir"].tolist(), df["labels_csv"].tolist())]

def topk_noisyor(P_seq: torch.Tensor, k: int = 10) -> torch.Tensor:
    """P_seq: T x E -> 1 x E using top-K frames per event."""
    k = min(k, P_seq.shape[0])
    vals, _ = torch.topk(P_seq, k, dim=0)     # k x E
    S = 1.0 - torch.clamp(1.0 - vals, min=1e-6).prod(dim=0)  # E
    return S.unsqueeze(0)                      # 1 x E

def build_ontology_priors(device: str):
    """Returns a mask M_dir (C x E) with 1 for atoms mapped to event e, else 0."""
    atom_to_idx = {a: i for i, a in enumerate(sorted(ALL_ATOMS))}
    C = len(atom_to_idx); E = len(EVENTS)
    M_dir = torch.zeros(C, E, device=device)
    for e_idx, e in enumerate(EVENTS):
        for a in EVENT2ATOMS[e]:
            a = a.lower()
            if a in atom_to_idx:
                M_dir[atom_to_idx[a], e_idx] = 1.0
    return M_dir  # encourage rho_dir towards 1 on these cells (if prior_w>0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True,
                    help="CSV with columns: run_dir,labels_csv (one row per video)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--freeze_bias_epochs", type=int, default=5)
    ap.add_argument("--rho_l1", type=float, default=1e-4, help="L1 sparsity on theta_*")
    ap.add_argument("--prior_w", type=float, default=1e-3, help="weight for ontology prior")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = build_manifest(args.manifest_csv)

    # Load first run to get C
    pbar0 = load_run(items[0][0], device)
    T0, C = pbar0.shape
    E = len(EVENTS)

    model = RoleLearner(C_atoms=C, E_events=E).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Priors
    M_dir = build_ontology_priors(device)  # C x E

    for ep in range(args.epochs):
        print('epoch: ' + str(ep))
        model.train()
        # Optional warmup: freeze bias & gammas so ρ learns first
        if ep < args.freeze_bias_epochs:
            for n, p in model.named_parameters():
                if n in ["gamma_dir", "gamma_ind", "gamma_ctr", "bias"]:
                    p.requires_grad_(False)
        else:
            for _, p in model.named_parameters():
                p.requires_grad_(True)

        total_loss = torch.tensor(0.0, device=device)
        n_vid = 0

        for run_dir, labels_csv in items:
            pbar = load_run(run_dir, device)      # T x C
            Y = load_Y(labels_csv, device)        # 1 x E

            P = model(pbar)                       # T x E
            S = topk_noisyor(P, k=args.topk)      # 1 x E
            bce = loss_video_level(S, Y)

            # Regularizers
            reg = torch.tensor(0.0, device=device)
            # L1 sparsity on theta_* (promote selective atoms)
            if hasattr(model, "theta_dir"):
                reg = reg + args.rho_l1 * (model.theta_dir.abs().mean()
                                           + model.theta_ind.abs().mean()
                                           + model.theta_ctr.abs().mean())
            else:
                # if using original version without theta-param, L1 on rho_dir/ind/ctr
                reg = reg + args.rho_l1 * (model.rho_dir.abs().mean()
                                           + model.rho_ind.abs().mean()
                                           + model.rho_ctr.abs().mean())

            # Ontology prior: encourage atoms mapped to event e to have higher direct gate
            # ρ_dir ≈ sigmoid(theta_dir)
            if hasattr(model, "theta_dir"):
                rho_dir = torch.sigmoid(model.theta_dir)
            else:
                rho_dir = model.rho_dir.clamp(0, 1)
            # BCE towards 1 on mapped entries; towards 0 on non-mapped (mild)
            eps = 1e-6
            target = M_dir
            prior = -(target * torch.log(rho_dir.clamp(eps,1-eps))
                      + (1 - target) * torch.log((1 - rho_dir).clamp(eps,1-eps))).mean()
            reg = reg + args.prior_w * prior

            loss = bce + reg
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach()
            n_vid += 1

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1:03d} | avg_loss={(total_loss / max(n_vid,1)).item():.4f} | videos={n_vid}")

    # Save
    out_dir = os.path.dirname(items[0][0]) if items else "."
    torch.save(model.state_dict(), os.path.join(out_dir, "role_learner_multi.pt"))
    print("Saved model to", os.path.join(out_dir, "role_learner_multi.pt"))

if __name__ == "__main__":
    main()
