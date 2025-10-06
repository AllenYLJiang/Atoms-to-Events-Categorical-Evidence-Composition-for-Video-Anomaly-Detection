import torch, itertools
from typing import Dict, List, Tuple
from ..atoms.ontology import EVENTS
import torch.nn.functional as F

class RoleLearner(torch.nn.Module):
    """
    Learns per-event role gates ρ^{dir}_{c,e}, ρ^{ind}_{c,e}, ρ^{ctr}_{c,e}
    and mixing weights γ_dir, γ_ind, γ_ctr, plus bias b_e.
    """
    def __init__(self, C_atoms: int, E_events: int):
        super().__init__()
        self.rho_dir = torch.nn.Parameter(0.5*torch.ones(C_atoms, E_events))
        self.rho_ind = torch.nn.Parameter(0.5*torch.ones(C_atoms, E_events))
        self.rho_ctr = torch.nn.Parameter(0.5*torch.ones(C_atoms, E_events))
        self.gamma_dir = torch.nn.Parameter(torch.ones(E_events))
        self.gamma_ind = torch.nn.Parameter(torch.ones(E_events))
        # Counter usually *reduces* probability -> start negative
        self.gamma_ctr = torch.nn.Parameter(-1.0*torch.ones(E_events))
        self.bias = torch.nn.Parameter(torch.zeros(E_events))

    def forward(self, pbar: torch.Tensor) -> torch.Tensor:
        """
        pbar: T x C aggregated presence \bar p_c(u)
        returns: T x E probabilities P_seg(e | u)
        """
        # Ensure inputs live on the same device as model params
        dev = next(self.parameters()).device
        pbar = pbar.to(dev)

        T, C = pbar.shape
        E = self.bias.numel()

        # Use local views of params on the same device (optional but clear)
        rho_dir = self.rho_dir.clamp(0, 1)
        rho_ind = self.rho_ind.clamp(0, 1)
        gamma_dir = self.gamma_dir
        gamma_ind = self.gamma_ind
        gamma_ctr = self.gamma_ctr
        bias = self.bias

        # Direct: D_e = 1 − Π_c (1 − ρ_dir_{c,e} · pbar_c)
        D_list = []
        one = torch.tensor(1.0, device=dev)
        eps = 1e-6
        for e in range(E):
            term = one - (one - (rho_dir[:, e] * pbar)).clamp(min=eps).prod(dim=1)
            D_list.append(term)
        D = torch.stack(D_list, dim=1)  # T x E

        # Indirect: I_e = 1 − Π_{i<j} (1 − ρ_i ρ_j p_i p_j)
        I_list = []
        for e in range(E):
            rho = rho_ind[:, e]
            terms = []
            # build products on the right device
            for i in range(C):
                for j in range(i + 1, C):
                    terms.append(one - (rho[i] * rho[j] * pbar[:, i] * pbar[:, j]))
            if len(terms) == 0:
                I_list.append(torch.zeros(T, device=dev))
            else:
                prod = torch.stack(terms, dim=1).clamp(min=eps).prod(dim=1)
                I_list.append(one - prod)
        I = torch.stack(I_list, dim=1)  # T x E

        # Counter evidence: no strong direct AND not two-indirects
        Ctr = (one - D) * (one - I)  # T x E

        # Combine in logit space
        def logit(x):
            x = x.clamp(eps, 1 - eps)
            return torch.log(x) - torch.log(1 - x)

        logits = (gamma_dir * logit(D)) + (gamma_ind * logit(I)) + (gamma_ctr * logit(Ctr)) + bias
        P = torch.sigmoid(logits)
        return P  # T x E


def video_union_probability(P_seq: torch.Tensor, U: List[List[int]]) -> torch.Tensor:
    """
    P_seq: (T x E) on some device
    U: list of lists of frame indices
    Returns: (N x E) on the same device as P_seq
    """
    dev = P_seq.device
    outs = []
    for idxs in U:
        probs = P_seq[idxs]                                      # (|U_n| x E) on dev
        S = 1.0 - (1.0 - probs).clamp(min=1e-6).prod(dim=0)      # (E,) on dev
        outs.append(S)
    return torch.stack(outs, dim=0)

def loss_video_level(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy over video-level event presence.
    S: (N x E) predicted probs, Y: (N x E) {0,1}
    """
    Y = Y.to(S.device)
    return F.binary_cross_entropy(S.clamp(1e-6, 1-1e-6), Y)
