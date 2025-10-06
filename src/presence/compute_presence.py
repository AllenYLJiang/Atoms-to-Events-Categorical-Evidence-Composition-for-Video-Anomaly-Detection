import torch
from typing import Dict, List, Tuple

def compute_alpha(p: torch.Tensor) -> torch.Tensor:
    """
    p: T x C presence per frame
    returns α_t = p[t]
    """
    return p

def inter_frame_divergence(alpha: torch.Tensor) -> torch.Tensor:
    """
    d_t = ||α_t − α_{t−1}||_2^2 for t>=1, with d_0=0
    """
    diff = alpha[1:] - alpha[:-1]
    d = (diff.pow(2).sum(dim=-1))
    d = torch.cat([torch.zeros(1, device=alpha.device), d], dim=0)
    return d

def boundary_prob(d: torch.Tensor, delta: float, kappa: float) -> torch.Tensor:
    """
    b_t = σ( κ ( d_t − δ ) ), (learnable δ, κ possible)
    """
    return torch.sigmoid(kappa * (d - delta))

def pi_t_to_u(b: torch.Tensor, t: int, u: int) -> torch.Tensor:
    """
    π_{t→u} = Π_{s=t+1}^u (1 − b_s)
    """
    if u <= t:
        return torch.tensor(1.0, device=b.device)
    segment = (1.0 - b[t+1:u+1])
    return segment.clamp(min=1e-6).prod()

def segment_weighted_presence(p: torch.Tensor, b: torch.Tensor, window: int) -> torch.Tensor:
    """
    \bar p_c(u) = sum_{t<=u, |u-t|<S_l} π_{t→u} * mean_{s=t..u} p_c(s)
    Returns T x C
    """
    T, C = p.shape
    out = torch.zeros_like(p)
    for u in range(T):
        t0 = max(0, u - window + 1)
        acc = torch.zeros(C, device=p.device)
        for t in range(t0, u+1):
            pi = pi_t_to_u(b, t, u)
            avg = p[t:u+1].mean(dim=0)
            acc = acc + pi * avg
        out[u] = acc
    return out
