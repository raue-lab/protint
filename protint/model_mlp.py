import torch
import torch.nn as nn
from typing import Union, Callable

# reuse your existing factories
from protint.model_cvae import make_activation, make_norm

ActivationSpec = Union[str, nn.Module, Callable[[], nn.Module]]
NormSpec = Union[str, Callable[[int], nn.Module], None]


class MLP(nn.Module):
    """
    Adversarial classifier on latent z to predict batch labels.

    Architecture is fixed (3 FC layers) but activation / normalization
    are fully configurable.
    """
    def __init__(
        self,
        enc_dim: int,
        output_dim: int,
        activation: ActivationSpec = "softplus",
        norm: NormSpec = "layernorm",
        norm_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        gn_groups: int = 8,
    ):
        super().__init__()

        self.activation = make_activation(activation)

        self.fc1 = nn.Linear(enc_dim, enc_dim)
        self.norm1 = make_norm(norm, enc_dim, eps=norm_eps,
                               bn_momentum=bn_momentum, gn_groups=gn_groups)

        self.fc2 = nn.Linear(enc_dim, enc_dim)
        self.norm2 = make_norm(norm, enc_dim, eps=norm_eps,
                               bn_momentum=bn_momentum, gn_groups=gn_groups)

        self.fc3 = nn.Linear(enc_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, z: torch.Tensor):
        h = self.fc1(z)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.fc2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        logits = self.fc3(h)
        return self.log_softmax(logits)