import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import NLLLoss


def reconstruction_gaussian_nll(
    x_mu: torch.Tensor,
    x_logvar: torch.Tensor,
    x: torch.Tensor,
    detected: torch.Tensor,
    eps: float = 1e-8,
    full: bool = True,
):
    """
    x_mu, x_logvar, x, detected: [B, P]
    Returns: scalar
    Sum over proteins, then sum over batch.
    """
    mask = detected.to(x_mu.dtype)
    var = torch.exp(x_logvar) + eps

    loss_fn = nn.GaussianNLLLoss(full=full, eps=eps, reduction="none")
    per_entry = loss_fn(x_mu, x, var)          # [B, P]

    return (per_entry * mask).sum()


def kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    eps = 1e-4
):
    """
    mu, logvar: [B, K]
    Returns: scalar
    Sum over latent dimensions, then sum over batch.
    """
    stdev = torch.exp(logvar) + eps
    prior = Normal(torch.zeros_like(mu), torch.ones_like(stdev))
    post = Normal(mu, stdev)
    return kl_divergence(post, prior).sum()


def dropout_loss(
    p_miss: torch.Tensor,
    detected: torch.Tensor,
):
    """
    p_obs, detected: [B, P]
    Returns: scalar
    Sum over proteins, then sum over batch.
    """
    target = 1 - detected.to(p_miss.dtype)
    return F.binary_cross_entropy(p_miss, target, reduction="none").sum()


def adversarial_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_weights,
):
    """
    pred: [B, C] log-probs
    target: [B, C] one-hot
    Returns: scalar
    Sum over batch.
    """
    if class_weights is not None:
        class_weights = class_weights.to(pred.device, pred.dtype)

    loss_function = NLLLoss(weight=class_weights, reduction="none")
    return loss_function(pred, torch.argmax(target, dim=1)).sum()   # [B]


def cycle_consistency_loss(
    z_original: torch.Tensor,
    z_switch: torch.Tensor,
    eps: float = 1e-8,
):
    """
    z_original, z_switch: [B, K]
    Returns: scalar
    Sum over latent dimensions, then sum over batch.
    """
    z_concat = torch.cat([z_original, z_switch], dim=0)
    z_mean = z_concat.mean(dim=0, keepdim=True)
    z_std = z_concat.std(dim=0, keepdim=True, unbiased=False) + eps

    z_orig_transformed = (z_original - z_mean) / z_std
    z_switch_transformed = (z_switch - z_mean) / z_std

    return F.mse_loss(
        z_orig_transformed,
        z_switch_transformed,
        reduction="none",
    ).sum()  # [B, K]
