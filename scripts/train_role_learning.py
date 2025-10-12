#!/usr/bin/env python3
import os, argparse
import torch, pandas as pd
from typing import List, Tuple
from src.train.role_learning import RoleLearner, loss_video_level  # keep your improved RoleLearner (sigmoid + log-domain)
import numpy as np
import pandas as pd
import torch
from src.atoms.ontology import EVENTS, ALL_ATOMS, EVENT2ATOMS

device = "cuda" if torch.cuda.is_available() else "cpu"

BIN_MAP = {
    "1": 1, "0": 0,
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "y": 1, "n": 0,
    "t": 1, "f": 0,
}

atoms_idx = {a:i for i,a in enumerate(sorted(ALL_ATOMS))}
C, E = len(atoms_idx), len(EVENTS)

M = torch.zeros(C, E, dtype=torch.bool)
for e_idx, e in enumerate(EVENTS):
    for a in EVENT2ATOMS[e]:   # "related" atoms; no role info needed
        if a in atoms_idx:
            M[atoms_idx[a], e_idx] = True
M = M.to(device)

def load_run(run_dir: str, device: str):
    pack = torch.load(os.path.join(run_dir, "trainpack.pt"), map_location=device)
    return pack["pbar"].to(device)  # (T, C)

def load_Y(video_labels_csv: str, device: str, events: list[str]) -> torch.Tensor:
    df = pd.read_csv(video_labels_csv)
    # schema check
    if list(df.columns[1:]) != events:
        raise ValueError(f"Columns must be ['video_id'] + {events}, got {list(df.columns)}")

    # take first row of event columns (shape E,)
    s = df.iloc[0, 1:]

    # 1) try numeric first (handles '0','1', 0,1, NaN)
    numeric = pd.to_numeric(s, errors="coerce")

    # 2) where numeric failed, try common string booleans
    need_map = numeric.isna()
    if need_map.any():
        mapped = (
            s.astype(str)
             .str.strip()
             .str.lower()
             .map(BIN_MAP)
             .astype("float32")
        )
        numeric[need_map] = mapped[need_map]

    # 3) fill any remaining NaNs with 0 (or raise if you prefer strictness)
    if numeric.isna().any():
        # Option A (strict): raise
        # bad = numeric[numeric.isna()]
        # raise ValueError(f"Non-parsable labels: {bad.index.tolist()}")
        # Option B (forgiving): default to 0
        numeric = numeric.fillna(0.0)

    vals = numeric.astype("float32").to_numpy()
    return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)  # (1, E)

def build_manifest(manifest_csv: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(manifest_csv)
    assert {"run_dir", "labels_csv"}.issubset(df.columns)
    return list(zip(df["run_dir"].tolist(), df["labels_csv"].tolist()))

def topk_noisyor(P_seq: torch.Tensor, k: int = 10, frac: float = 0.02, cap: float = 0.9, tau: float = 2.0):
    """
    P_seq: (T, E). Use top-k where k = max(1, min(k, ceil(frac*T))).
    Shrink via temperature before union; cap per-frame probs to cap (<1).
    """
    T, E = P_seq.shape
    k_eff = max(1, min(k, int((T*frac) + 0.5)))
    vals, _ = torch.topk(P_seq, k_eff, dim=0)                  # (k_eff, E)
    eps = 1e-6
    def logit(x): x = x.clamp(eps, 1-eps); return torch.log(x) - torch.log(1-x)
    vals = torch.sigmoid(logit(vals) / tau).clamp(max=cap)     # cooled & capped
    S = 1.0 - torch.clamp(1.0 - vals, min=eps).prod(0)         # (E,)
    return S.unsqueeze(0)

def main():
    from src.atoms.ontology import EVENTS  # only to know E and to validate CSV order
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--freeze_bias_epochs", type=int, default=5)
    args = ap.parse_args()

    items = build_manifest(args.manifest_csv)
    assert items, "Empty manifest."

    # infer C from first run
    pbar0 = load_run(items[0][0], device)
    T0, C = pbar0.shape
    E = len(EVENTS)

    model = RoleLearner(C_atoms=C, E_events=E).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()

        # warm-up: learn rho first
        if ep < args.freeze_bias_epochs:
            for n, p in model.named_parameters():
                if n in ["gamma_dir","gamma_ind","gamma_ctr","bias"]:
                    p.requires_grad_(False)
        else:
            for _, p in model.named_parameters():
                p.requires_grad_(True)

        total_loss = torch.tensor(0.0, device=device)
        n_vid = 0

        for run_dir, labels_csv in items:
            pbar = load_run(run_dir, device)                         # (T, C)
            Y = load_Y(labels_csv, device, EVENTS)                   # (1, E)

            # P = model(pbar)                                        # (T, E)
            P = model(pbar, candidate_mask=M)                          # (T, E)
            S = topk_noisyor(P, k=args.topk, frac=0.02, cap=0.9, tau=2.0)                         # (1, E)
            loss = loss_video_level(S, Y)                            # BCE

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach()
            n_vid += 1

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1:03d} | avg_loss={(total_loss / max(n_vid,1)).item():.4f} | videos={n_vid}")

    # save model next to first run (or choose your own path)
    out_dir = os.path.dirname(items[0][0])
    torch.save(model.state_dict(), os.path.join(out_dir, "role_learner_multi.pt"))
    print("Saved:", os.path.join(out_dir, "role_learner_multi.pt"))

if __name__ == "__main__":
    main()
