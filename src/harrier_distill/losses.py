"""Distillation losses: pad-safe KL matrix, focal InfoNCE, disperse, STS MSE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    """x: [N, D] -> [N, N] cosine similarities (rows L2-normalized)."""
    x = F.normalize(x, p=2, dim=-1)
    return x @ x.T


def _valid_pair_mask(n: int, valid: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """True where both i and j are valid and i != j."""
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=device)
    else:
        valid = valid.bool().to(device)
    eye = torch.eye(n, device=device, dtype=torch.bool)
    return valid.unsqueeze(0) & valid.unsqueeze(1) & ~eye


def kl_similarity_matrix(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    temperature: float = 0.07,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(softmax(S_T/τ) || softmax(S_S/τ)) over valid rows only; pads masked with -inf."""
    n = student_emb.size(0)
    if n < 2:
        return student_emb.new_zeros(())

    device = student_emb.device
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=device)
    else:
        valid = valid.bool().to(device)

    if valid.sum() < 2:
        return student_emb.new_zeros(())

    s_s = pairwise_cosine(student_emb) / temperature
    s_t = pairwise_cosine(teacher_emb) / temperature

    # Mask invalid columns so softmax ignores pads; also zero invalid rows later
    inv_col = ~valid.unsqueeze(0).expand(n, n)
    s_s = s_s.masked_fill(inv_col, float("-inf"))
    s_t = s_t.masked_fill(inv_col, float("-inf"))
    # Self should not dominate: keep diagonal finite but we still softmax over others+self;
    # set diagonal to -inf among valid rows for a leave-one-out style neighborhood.
    eye = torch.eye(n, device=device, dtype=torch.bool)
    s_s = s_s.masked_fill(eye, float("-inf"))
    s_t = s_t.masked_fill(eye, float("-inf"))

    log_p_s = F.log_softmax(s_s, dim=-1)
    p_t = F.softmax(s_t, dim=-1)
    # Replace NaNs from all-masked rows
    log_p_s = torch.nan_to_num(log_p_s, nan=0.0, posinf=0.0, neginf=0.0)
    p_t = torch.nan_to_num(p_t, nan=0.0)

    # Per-row KL, average over valid rows only
    kl_rows = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
    kl_rows = kl_rows[valid]
    return kl_rows.mean() if kl_rows.numel() else student_emb.new_zeros(())


def focal_infonce(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
    gamma: float = 2.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    InfoNCE with focal weights on hard negatives from teacher similarities.
    Skips invalid/pad anchors and candidates.
    """
    n = student_emb.size(0)
    if n < 2:
        return student_emb.new_zeros(())

    device = student_emb.device
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=device)
    else:
        valid = valid.bool().to(device)

    logits = pairwise_cosine(student_emb) / temperature
    with torch.no_grad():
        t_sim = pairwise_cosine(teacher_emb)

    eye = torch.eye(n, device=device, dtype=torch.bool)
    pos = positive_mask.bool() & ~eye
    # Drop pairs involving pads
    pos = pos & valid.unsqueeze(0) & valid.unsqueeze(1)
    if not pos.any():
        return student_emb.new_zeros(())

    losses = []
    for i in range(n):
        if not bool(valid[i]):
            continue
        pos_idx = pos[i].nonzero(as_tuple=False).view(-1)
        if pos_idx.numel() == 0:
            continue
        neg_mask = ~eye[i] & ~pos[i] & valid
        neg_idx = neg_mask.nonzero(as_tuple=False).view(-1)
        if neg_idx.numel() == 0:
            continue

        neg_teacher = t_sim[i, neg_idx]
        difficulty = ((neg_teacher + 1.0) * 0.5).clamp(0, 1)
        weights = difficulty.pow(gamma)
        weights = weights / (weights.sum() + 1e-8)

        for j in pos_idx.tolist():
            pos_logit = logits[i, j]
            neg_logits = logits[i, neg_idx]
            max_logit = torch.maximum(pos_logit, neg_logits.max())
            pos_exp = torch.exp(pos_logit - max_logit)
            neg_exp = torch.exp(neg_logits - max_logit)
            denom = pos_exp + (weights * neg_exp).sum()
            losses.append(-torch.log(pos_exp / (denom + 1e-8)))

    if not losses:
        return student_emb.new_zeros(())
    return torch.stack(losses).mean()


def disperse_loss(
    student_emb: torch.Tensor,
    t: float = 2.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Uniformity / disperse over valid off-diagonal pairs only."""
    n = student_emb.size(0)
    if n < 2:
        return student_emb.new_zeros(())
    pair_ok = _valid_pair_mask(n, valid, student_emb.device)
    if not pair_ok.any():
        return student_emb.new_zeros(())
    sim = pairwise_cosine(student_emb)
    off = sim.masked_select(pair_ok)
    return torch.log(torch.exp(t * off).mean() + 1e-8)


def sts_pair_cosine_mse(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    pair_index: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MSE between student and teacher pairwise cosines for listed pairs.

    pair_index: LongTensor [P, 2] of (i, j) indices into the emb batch.
    """
    if pair_index is None or pair_index.numel() == 0:
        return student_emb.new_zeros(())
    if pair_index.dim() != 2 or pair_index.size(-1) != 2:
        raise ValueError(f"pair_index must be [P,2], got {tuple(pair_index.shape)}")

    device = student_emb.device
    pairs = pair_index.to(device=device, dtype=torch.long)
    n = student_emb.size(0)
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=device)
    else:
        valid = valid.bool().to(device)

    i = pairs[:, 0]
    j = pairs[:, 1]
    keep = (i >= 0) & (j >= 0) & (i < n) & (j < n) & valid[i] & valid[j] & (i != j)
    if not keep.any():
        return student_emb.new_zeros(())
    i, j = i[keep], j[keep]

    s = F.normalize(student_emb, p=2, dim=-1)
    t = F.normalize(teacher_emb, p=2, dim=-1)
    cos_s = (s[i] * s[j]).sum(dim=-1)
    with torch.no_grad():
        cos_t = (t[i] * t[j]).sum(dim=-1)
    return F.mse_loss(cos_s, cos_t)


class DistillLossBundle(torch.nn.Module):
    def __init__(
        self,
        lambda_kl: float = 1.0,
        lambda_focal: float = 0.5,
        lambda_disp: float = 0.01,
        lambda_sts: float = 1.0,
        temperature: float = 0.07,
        focal_gamma: float = 2.0,
        disperse_t: float = 2.0,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_focal = lambda_focal
        self.lambda_disp = lambda_disp
        self.lambda_sts = lambda_sts
        self.temperature = temperature
        self.focal_gamma = focal_gamma
        self.disperse_t = disperse_t

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        positive_mask: torch.Tensor,
        valid: torch.Tensor | None = None,
        sts_pairs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_kl = kl_similarity_matrix(
            student_emb, teacher_emb, self.temperature, valid=valid
        )
        l_focal = focal_infonce(
            student_emb,
            teacher_emb,
            positive_mask,
            temperature=self.temperature,
            gamma=self.focal_gamma,
            valid=valid,
        )
        l_disp = disperse_loss(student_emb, t=self.disperse_t, valid=valid)
        l_sts = sts_pair_cosine_mse(student_emb, teacher_emb, sts_pairs, valid=valid)
        total = (
            self.lambda_kl * l_kl
            + self.lambda_focal * l_focal
            + self.lambda_disp * l_disp
            + self.lambda_sts * l_sts
        )
        return {
            "loss": total,
            "loss_kl": l_kl.detach(),
            "loss_focal": l_focal.detach(),
            "loss_disp": l_disp.detach(),
            "loss_sts": l_sts.detach(),
        }
