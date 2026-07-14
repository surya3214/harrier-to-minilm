"""Distillation losses: KL matrix, focal InfoNCE, disperse."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    """x: [N, D] L2-normalized -> [N, N] cosine similarities."""
    x = F.normalize(x, p=2, dim=-1)
    return x @ x.T


def kl_similarity_matrix(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """KL(softmax(S_T/τ) || softmax(S_S/τ)) averaged over rows."""
    s_s = pairwise_cosine(student_emb) / temperature
    s_t = pairwise_cosine(teacher_emb) / temperature
    log_p_s = F.log_softmax(s_s, dim=-1)
    p_t = F.softmax(s_t, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean")


def focal_infonce(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.05,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    InfoNCE with focal weights on hard negatives from teacher similarities.

    positive_mask: [N, N] bool/float, True where (i, j) is a positive pair (symmetric ok).
    Diagonal is ignored (self).
    """
    n = student_emb.size(0)
    if n < 2:
        return student_emb.new_zeros(())

    logits = pairwise_cosine(student_emb) / temperature
    with torch.no_grad():
        t_sim = pairwise_cosine(teacher_emb)

    # Exclude self
    eye = torch.eye(n, device=student_emb.device, dtype=torch.bool)
    pos = positive_mask.bool() & ~eye
    if not pos.any():
        return student_emb.new_zeros(())

    # For each anchor with at least one positive, average over its positives
    losses = []
    for i in range(n):
        pos_idx = pos[i].nonzero(as_tuple=False).view(-1)
        if pos_idx.numel() == 0:
            continue
        # negatives = all non-self, non-positive
        neg_mask = ~eye[i] & ~pos[i]
        neg_idx = neg_mask.nonzero(as_tuple=False).view(-1)
        if neg_idx.numel() == 0:
            continue

        # Focal weight: harder teacher negatives (higher t_sim) get larger weight
        neg_teacher = t_sim[i, neg_idx]
        # map similarity in [-1,1] roughly to (0,1) difficulty
        difficulty = ((neg_teacher + 1.0) * 0.5).clamp(0, 1)
        weights = difficulty.pow(gamma)
        weights = weights / (weights.sum() + 1e-8)

        for j in pos_idx.tolist():
            pos_logit = logits[i, j]
            neg_logits = logits[i, neg_idx]
            # weighted log-softmax style InfoNCE
            # L = -log( exp(pos) / (exp(pos) + sum_k w_k exp(neg_k)) )
            max_logit = torch.maximum(pos_logit, neg_logits.max())
            pos_exp = torch.exp(pos_logit - max_logit)
            neg_exp = torch.exp(neg_logits - max_logit)
            denom = pos_exp + (weights * neg_exp).sum()
            losses.append(-torch.log(pos_exp / (denom + 1e-8)))

    if not losses:
        return student_emb.new_zeros(())
    return torch.stack(losses).mean()


def disperse_loss(student_emb: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Uniformity / disperse: log mean exp(t * cosine) over off-diagonal pairs."""
    n = student_emb.size(0)
    if n < 2:
        return student_emb.new_zeros(())
    sim = pairwise_cosine(student_emb)
    eye = torch.eye(n, device=student_emb.device, dtype=torch.bool)
    off = sim.masked_select(~eye)
    return torch.log(torch.exp(t * off).mean() + 1e-8)


class DistillLossBundle(torch.nn.Module):
    def __init__(
        self,
        lambda_kl: float = 1.0,
        lambda_focal: float = 0.5,
        lambda_disp: float = 0.05,
        temperature: float = 0.05,
        focal_gamma: float = 2.0,
        disperse_t: float = 2.0,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_focal = lambda_focal
        self.lambda_disp = lambda_disp
        self.temperature = temperature
        self.focal_gamma = focal_gamma
        self.disperse_t = disperse_t

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_kl = kl_similarity_matrix(student_emb, teacher_emb, self.temperature)
        l_focal = focal_infonce(
            student_emb,
            teacher_emb,
            positive_mask,
            temperature=self.temperature,
            gamma=self.focal_gamma,
        )
        l_disp = disperse_loss(student_emb, t=self.disperse_t)
        total = (
            self.lambda_kl * l_kl
            + self.lambda_focal * l_focal
            + self.lambda_disp * l_disp
        )
        return {
            "loss": total,
            "loss_kl": l_kl.detach(),
            "loss_focal": l_focal.detach(),
            "loss_disp": l_disp.detach(),
        }
