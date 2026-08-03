# Model utils, copied from https://github.com/Novartis/MOBER/blob/main/mober/models/utils.py from the MOBER paper

import torch
import math
from typing import Sequence, Union, Callable, Optional, Dict
from torch import optim, nn
ActivationSpec = Union[str, nn.Module, Callable[[], nn.Module]]
NormSpec = Union[str, Callable[[int], nn.Module], None]


def create_model(model_cls, device, *args, filename=None, lr=1e-3, **kwargs):
    """
    Simple model serialization to resume training from given epoch.

    :param model_cls: Model definition
    :param device: Device (cpu or gpu)
    :param args: arguments to be passed to the model constructor
    :param filename: filename if the model is to be loaded
    :param lr: learning rate to be used by the model optimizer
    :param kwargs: keyword arguments to be used by the model constructor
    :return:
    """
    model = model_cls(*args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if filename is not None:
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded model epoch: {checkpoint['epoch']}, loss {checkpoint['loss']}")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Loading model on ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    return model.to(device), optimizer


def save_model(model, optimizer, epoch, loss, filename, device):
    """
    Save the model to a file.

    :param model: model to be saved
    :param optimizer: model optimizer
    :param epoch: number of epoch, only for information
    :param loss: loss, only for information
    :param filename: where to save the model
    :param device: device of the model
    """
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, filename)

def make_activation(act: ActivationSpec) -> nn.Module:
    if isinstance(act, nn.Module):
        return act
    if callable(act) and not isinstance(act, str):
        return act()

    if not isinstance(act, str):
        raise TypeError(f"Unsupported activation spec: {type(act)}")

    key = act.lower()
    if key == "softplus":
        return nn.Softplus()
    if key == "selu":
        return nn.SELU()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key in ("identity", "none", "linear"):
        return nn.Identity()

    raise ValueError(f"Unknown activation '{act}'")


def make_norm(
    norm: NormSpec,
    num_features: int,
    *,
    eps: float = 1e-5,
    bn_momentum: float = 0.1,
    gn_groups: int = 8,
) -> nn.Module:
    if norm is None:
        return nn.Identity()

    if callable(norm) and not isinstance(norm, str):
        return norm(num_features)

    if not isinstance(norm, str):
        raise TypeError(f"Unsupported norm spec: {type(norm)}")

    key = norm.lower()
    if key in ("layernorm", "ln"):
        return nn.LayerNorm(num_features, eps=eps)
    if key in ("batchnorm", "bn", "batchnorm1d"):
        return nn.BatchNorm1d(num_features, eps=eps, momentum=bn_momentum)
    if key in ("instancenorm", "in", "instancenorm1d"):
        return nn.InstanceNorm1d(
            num_features, eps=eps, momentum=bn_momentum, affine=True
        )
    if key in ("groupnorm", "gn"):
        if gn_groups <= 0:
            raise ValueError("gn_groups must be > 0")
        return nn.GroupNorm(gn_groups, num_features, eps=eps)

    if key in ("none", "identity", "no"):
        return nn.Identity()

    raise ValueError(f"Unknown norm: {norm}")


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def dropout_curve_from_mu(
    x_mu: torch.Tensor,
    rho: torch.Tensor,
    zeta: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Missingness curve:
        p_miss = Phi((rho - x_mu) / zeta)

    Shapes
    ------
    x_mu  : (B, D)
    rho   : (B, D), (D,), or (1, D)
    zeta  : (B, D), (D,), or (1, D)
    """
    if rho.dim() == 1:
        rho = rho.unsqueeze(0)
    if zeta.dim() == 1:
        zeta = zeta.unsqueeze(0)

    zeta = zeta.clamp_min(eps)
    p_miss = normal_cdf((rho - x_mu) / zeta)
    p_miss = p_miss.clamp(eps, 1.0 - eps)
    return p_miss
