# src/train/role_learning.py (replace RoleLearner with this version)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoleLearner(nn.Module):
    def __init__(self, C_atoms: int, E_events: int, p0_dir=0.02, p0_ind=0.01, p0_ctr=0.2,
                 logit_base=math.log(0.05/0.95),  # ~ 5% base prob
                 logit_clip=8.0, temperature=2.0):
        super().__init__()
        self.C = C_atoms
        self.E = E_events
        self.logit_clip = float(logit_clip)
        self.temperature = float(temperature)

        # Unconstrained parameters; rho = sigmoid(theta)
        self.theta_dir = nn.Parameter(torch.zeros(self.C, self.E))
        self.theta_ind = nn.Parameter(torch.zeros(self.C, self.E))
        self.theta_ctr = nn.Parameter(torch.zeros(self.C, self.E))

        # Mixing weights (start at 0 so they don't blow up logits)
        self.gamma_dir = nn.Parameter(torch.zeros(self.E))
        self.gamma_ind = nn.Parameter(torch.zeros(self.E))
        self.gamma_ctr = nn.Parameter(torch.zeros(self.E))

        # Bias: small negative ⇒ base prob ~ 5% at init
        self.bias = nn.Parameter(torch.full((self.E,), float(logit_base)))

        # Initialize thetas to tiny rhos
        with torch.no_grad():
            def inv_sigmoid(p):  # logit
                p = min(max(p, 1e-4), 1-1e-4)
                return math.log(p/(1-p))
            self.theta_dir.fill_(inv_sigmoid(p0_dir))
            self.theta_ind.fill_(inv_sigmoid(p0_ind))
            self.theta_ctr.fill_(inv_sigmoid(p0_ctr))

    def _rho(self):
        rho_dir = torch.sigmoid(self.theta_dir)
        rho_ind = torch.sigmoid(self.theta_ind)
        rho_ctr = torch.sigmoid(self.theta_ctr)
        return rho_dir, rho_ind, rho_ctr

    def forward(self, pbar: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        pbar: (T, C) segment-weighted presence in [0,1]
        candidate_mask (optional): (C, E) boolean mask of which atoms are *eligible*
                                   for each event; non-eligible atoms are ignored.
        returns: P (T, E) segment-level event probabilities
        """
        dev = next(self.parameters()).device
        pbar = pbar.to(dev)
        T, C = pbar.shape
        assert C == self.C

        rho_dir, rho_ind, _rho_ctr = self._rho()  # (C, E)

        # Optionally zero-out non-candidate atoms (prevents spurious inflation)
        if candidate_mask is not None:
            # p_masked: (T, C, E)
            mask = candidate_mask.to(dev).bool().unsqueeze(0).expand(T, -1, -1)
        else:
            mask = torch.ones(T, C, self.E, dtype=torch.bool, device=dev)

        # ==== DIRECT: D_e = 1 - Π_c (1 - rho_dir_{c,e} * pbar_c)  (log-domain) ====
        # (T, C, E): 1 - rho*p
        one_minus = 1.0 - (pbar.unsqueeze(2) * rho_dir.unsqueeze(0))
        one_minus = torch.where(mask, one_minus, torch.ones_like(one_minus))
        one_minus = one_minus.clamp(min=1e-6)
        log_prod = torch.sum(torch.log(one_minus), dim=1)   # (T, E)
        D = 1.0 - torch.exp(log_prod)                       # (T, E)

        # ==== INDIRECT: I_e = 1 - Π_{i<j} (1 - rho_i rho_j p_i p_j) ====
        # Vectorized pairwise (use mask to drop non-candidates early)
        p = pbar.clamp(0,1)
        rho = rho_ind.clamp(0,1)
        # (T, C, 1) and (T, 1, C)
        p_i = p.unsqueeze(2); p_j = p.unsqueeze(1)          # (T, C, C)
        # (C, 1, E) and (1, C, E)
        r_i = rho.unsqueeze(1); r_j = rho.unsqueeze(0)      # (C, C, E)
        term = 1.0 - (p_i * p_j).unsqueeze(3) * (r_i * r_j).unsqueeze(0)   # (T, C, C, E)

        # mask i>=j and non-candidates (at least one of the pair must be candidate)
        tri = torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=1)  # (C, C)
        pair_mask = tri.unsqueeze(0).unsqueeze(3).expand(T, -1, -1, self.E)          # (T, C, C, E)
        # candidate: if both atoms not candidates for e, drop pair
        cm = candidate_mask.to(dev).bool() if candidate_mask is not None else torch.ones(C, self.E, dtype=torch.bool, device=dev)
        # both-not-candidate mask
        both_not = (~cm.unsqueeze(0)) & (~cm.unsqueeze(1))   # (C, C, E)
        both_not = both_not.unsqueeze(0).expand(T, -1, -1, -1)
        term = torch.where(pair_mask & (~both_not), term, torch.ones_like(term))
        term = term.clamp(min=1e-6)
        log_prod_pairs = torch.sum(torch.log(term), dim=(1,2))             # (T, E)
        I = 1.0 - torch.exp(log_prod_pairs)                                # (T, E)

        # ==== COUNTER: Ctr = (1 - D) * (1 - I) ====
        Ctr = (1.0 - D) * (1.0 - I)

        # ==== Mix in logit space with temperature and clamped logits ====
        def logit(x):
            x = x.clamp(1e-6, 1-1e-6)
            return torch.log(x) - torch.log(1-x)

        logits = (self.gamma_dir * (logit(D) / self.temperature)
                  + self.gamma_ind * (logit(I) / self.temperature)
                  + self.gamma_ctr * (logit(Ctr) / self.temperature)
                  + self.bias)
        logits = logits.clamp(-self.logit_clip, self.logit_clip)
        P = torch.sigmoid(logits)  # (T, E)
        return P

def loss_video_level(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Y = Y.to(S.device).to(S.dtype)
    return F.binary_cross_entropy(S.clamp(1e-6, 1-1e-6), Y)
