import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sum_log_prob(prob, samples):
    """
    Compute the sum of log probability of samples under the given distribution
    """
    # size = [n_z_samples, batch_size, *]
    log_prob = prob.log_prob(samples)
    log_prob = torch.sum(log_prob, dim=2)
    return log_prob


class Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_outputs, y_target):
        """
        Compute the loss
        """
        p_yCc, z_samples, q_zCc, q_zCct = pred_outputs

        if self.training:
            loss, kl_z, negative_ll = self.get_loss(
                p_yCc, z_samples, q_zCc, q_zCct, y_target
            )

        else:
            raise (NotImplementedError)

        if self.reduction is None:
            return loss, kl_z, negative_ll
        elif self.reduction == "mean":
            return torch.mean(loss), torch.mean(kl_z), torch.mean(negative_ll)
        elif self.reduction == "sum":
            return torch.sum(loss), torch.sum(kl_z), torch.sum(negative_ll)
        else:
            raise (NotImplementedError)


class ELBOLoss(Loss):
    def __init__(self, reduction="mean", beta=1):
        super().__init__(reduction=reduction)
        self.beta = beta

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, y_target):
        """
        Compute the ELBO loss during training and NLLL for validation and testing
        """
        if q_zCct is not None:
            # 1st term: E_{q(z | T)}[p(y_t | z)]
            sum_log_p_yCz = sum_log_prob(p_yCc, y_target)  # [num_z_samples, batch_size]
            E_z_sum_log_p_yCz = torch.mean(sum_log_p_yCz, dim=0)  # [batch_size]
            # 2nd term: KL[q(z | C, T) || q (z || C)]
            kl_z = torch.distributions.kl.kl_divergence(
                q_zCct, q_zCc
            )  # [batch_size, *n_lat]
            E_z_kl = torch.sum(kl_z, dim=1)  # [batch_size]
            loss = -(E_z_sum_log_p_yCz - self.beta * E_z_kl)
            negative_ll = -E_z_sum_log_p_yCz

        else:
            sum_log_p_yCz = sum_log_prob(p_yCc, y_target)
            sum_log_w_k = sum_log_p_yCz
            log_S_z_sum_p_y_Cz = torch.logsumexp(sum_log_w_k, 0)
            log_E_z_sum_p_yCz = log_S_z_sum_p_y_Cz - math.log(sum_log_w_k.shape[0])
            kl_z = torch.zeros_like(log_E_z_sum_p_yCz)
            negative_ll = -log_E_z_sum_p_yCz
            loss = negative_ll

        return loss, kl_z, negative_ll


class NLL(Loss):
    """
    Compute the approximate negative log likelihood
    """

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, y_target):
        sum_log_p_yCz = sum_log_prob(p_yCc, y_target)
        # importance sampling:
        if q_zCct is not None:
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
        else:
            sum_log_w_k = sum_log_p_yCz

        log_s_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)

        log_E_z_sum_p_yCz = log_s_z_sum_p_yCz - math.log(sum_log_w_k.shape[0])

        return (
            -log_E_z_sum_p_yCz,
            torch.zeros_like(log_E_z_sum_p_yCz),
            -log_E_z_sum_p_yCz,
        )


class InfoNCEAlignmentLoss(nn.Module):
    """Symmetric InfoNCE to align knowledge embedding k with task representation r."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, k, r):
        """
        Args:
            k: [bs, 1, D] knowledge embeddings
            r: [bs, 1, D] task representations (rC or rT mean)
        Returns:
            dict with alignment_loss, retrieval_acc, mean_cosine, n_valid_pairs
        """
        bs = k.shape[0]
        k = k.reshape(bs, -1)  # [bs, D]
        r = r.reshape(bs, -1)  # [bs, D]

        if bs < 2:
            zero = torch.tensor(0.0, device=k.device)
            return {
                "alignment_loss": zero,
                "retrieval_acc": zero,
                "mean_cosine": zero,
                "n_valid_pairs": 0,
            }

        k_norm = F.normalize(k, dim=-1)
        r_norm = F.normalize(r, dim=-1)

        logits = torch.mm(k_norm, r_norm.t()) / self.temperature  # [bs, bs]
        labels = torch.arange(bs, device=k.device)

        loss_kr = F.cross_entropy(logits, labels)
        loss_rk = F.cross_entropy(logits.t(), labels)
        alignment_loss = (loss_kr + loss_rk) / 2.0

        with torch.no_grad():
            retrieval_acc = (logits.argmax(dim=1) == labels).float().mean()
            mean_cosine = (k_norm * r_norm).sum(dim=-1).mean()

        return {
            "alignment_loss": alignment_loss,
            "retrieval_acc": retrieval_acc,
            "mean_cosine": mean_cosine,
            "n_valid_pairs": bs,
        }
