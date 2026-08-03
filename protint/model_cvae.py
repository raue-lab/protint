
from typing import Sequence, Union, Callable, Optional, Dict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

ActivationSpec = Union[str, nn.Module, Callable[[], nn.Module]]
NormSpec = Union[str, Callable[[int], nn.Module], None]

from protint.model_utils import make_activation, make_norm, normal_cdf, dropout_curve_from_mu

class Encoder(nn.Module):
    """
    Encoder with batch conditioning by concatenating a batch embedding.
    batch is assumed to be one-hot float tensor of shape (B, n_batch).
    """
    def __init__(
        self,
        n_genes: int,
        enc_dim: int,
        n_batch: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout_p: float = 0.1,
        activation: ActivationSpec = "softplus",
        norm: NormSpec = "layernorm",
        norm_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        gn_groups: int = 8,
        batch_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.activation = make_activation(activation)

        self.batch_embed_dim = batch_embed_dim if batch_embed_dim is not None else n_batch
        self.batch_fc = nn.Linear(n_batch, self.batch_embed_dim)
        self.batch_norm = make_norm(
            norm,
            self.batch_embed_dim,
            eps=norm_eps,
            bn_momentum=bn_momentum,
            gn_groups=gn_groups,
        )

        dims = [n_genes] + list(hidden_dims)
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dps = nn.ModuleList()

        for _, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            in_features = in_d + self.batch_embed_dim

            self.fcs.append(nn.Linear(in_features, out_d))
            self.norms.append(
                make_norm(
                    norm,
                    out_d,
                    eps=norm_eps,
                    bn_momentum=bn_momentum,
                    gn_groups=gn_groups,
                )
            )
            self.dps.append(nn.Dropout(p=dropout_p))

        last_h = dims[-1] if len(dims) > 1 else n_genes
        self.linear_means = nn.Linear(last_h, enc_dim)
        self.linear_log_vars = nn.Linear(last_h, enc_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        b = self.activation(self.batch_norm(self.batch_fc(batch)))

        h = x
        for _, (fc, nm, dp) in enumerate(zip(self.fcs, self.norms, self.dps)):
            h = torch.cat([h, b], dim=1)
            h = fc(h)
            h = nm(h)
            h = self.activation(h)
            h = dp(h)

        z_mu = self.linear_means(h)
        z_logvar = self.linear_log_vars(h)
        z_std = torch.exp(0.5 * z_logvar) + 1e-4
        z = Normal(z_mu, z_std).rsample()
        return z_mu, z_logvar, z


class DecoderBackbone(nn.Module):
    """
    Shared decoder backbone with batch conditioning.
    """
    def __init__(
        self,
        enc_dim: int,
        n_batch: int,
        hidden_dims: Sequence[int] = (128, 64),
        activation: ActivationSpec = "softplus",
        norm: NormSpec = "layernorm",
        norm_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        gn_groups: int = 8,
        batch_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.activation = make_activation(activation)

        self.batch_embed_dim = batch_embed_dim if batch_embed_dim is not None else n_batch
        self.batch_fc = nn.Linear(n_batch, self.batch_embed_dim)
        self.batch_norm = make_norm(
            norm,
            self.batch_embed_dim,
            eps=norm_eps,
            bn_momentum=bn_momentum,
            gn_groups=gn_groups,
        )

        dec_hidden = list(hidden_dims)[::-1]
        dims = [enc_dim] + dec_hidden

        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            in_features = in_d + self.batch_embed_dim

            self.fcs.append(nn.Linear(in_features, out_d))
            self.norms.append(
                make_norm(
                    norm,
                    out_d,
                    eps=norm_eps,
                    bn_momentum=bn_momentum,
                    gn_groups=gn_groups,
                )
            )

        self.out_dim = dims[-1] if len(dims) > 1 else enc_dim

    def forward(self, z: torch.Tensor, batch: torch.Tensor):
        b = self.activation(self.batch_norm(self.batch_fc(batch)))

        h = z
        for _, (fc, nm) in enumerate(zip(self.fcs, self.norms)):
            h = torch.cat([h, b], dim=1)
            h = fc(h)
            h = nm(h)
            h = self.activation(h)

        return h


class IntensityHead(nn.Module):
    """
    Decoder head for Gaussian intensity model.
    Outputs mean and log-variance per protein.
    """
    def __init__(self, in_dim: int, n_genes: int):
        super().__init__()
        self.mu_head = nn.Linear(in_dim, n_genes)
        self.logvar_head = nn.Linear(in_dim, n_genes)

    def forward(self, h: torch.Tensor):
        x_mu = self.mu_head(h)
        x_logvar = self.logvar_head(h)
        return x_mu, x_logvar


class BatchCVAEFeatureWiseDropout(nn.Module):
    """
    Conditional VAE with:
      - encoder conditioned on batch
      - decoder conditioned on batch
      - Gaussian output head: x_mu, x_logvar
      - feature-wise dropout parameters rho_d, zeta_d shared across samples

    Notes
    -----
    x passed into forward() should not contain NaNs.
    Use the raw x + obs_mask in the loss function.
    """
    def __init__(
        self,
        n_genes: int,
        enc_dim: int,
        n_batch: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout_p: float = 0.1,
        activation: ActivationSpec = "softplus",
        norm: NormSpec = "batchnorm",
        norm_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        gn_groups: int = 8,
        batch_embed_dim: Optional[int] = None,
        dropout_params_df: pd.DataFrame = None,
        train_dropout: bool = False
    ):
        super().__init__()
        self.n_genes = n_genes

        self.encoder = Encoder(
            n_genes=n_genes,
            enc_dim=enc_dim,
            n_batch=n_batch,
            hidden_dims=hidden_dims,
            dropout_p=dropout_p,
            activation=activation,
            norm=norm,
            norm_eps=norm_eps,
            bn_momentum=bn_momentum,
            gn_groups=gn_groups,
            batch_embed_dim=batch_embed_dim,
        )

        self.decoder_backbone = DecoderBackbone(
            enc_dim=enc_dim,
            n_batch=n_batch,
            hidden_dims=hidden_dims,
            activation=activation,
            norm=norm,
            norm_eps=norm_eps,
            bn_momentum=bn_momentum,
            gn_groups=gn_groups,
            batch_embed_dim=batch_embed_dim,
        )
        
        # output for intensities
        self.intensity_head = IntensityHead(self.decoder_backbone.out_dim, n_genes)

        if dropout_params_df is not None:
            rho_batch = torch.as_tensor(
                dropout_params_df["rho"].to_numpy(),
                dtype=torch.float32,
            ).view(n_batch, 1)

            zeta_batch = torch.as_tensor(
                dropout_params_df["zeta"].to_numpy(),
                dtype=torch.float32,
            ).view(n_batch, 1)

            rho0 = rho_batch.repeat(1, n_genes)
            zeta0 = zeta_batch.repeat(1, n_genes)
        else:
            # placeholder when loading the model from a checkpoint
            rho0 = torch.zeros((n_batch, n_genes), dtype=torch.float32)
            zeta0 = torch.ones((n_batch, n_genes), dtype=torch.float32)

        zeta0 = zeta0.clamp_min(1e-4)
        log_zeta0 = torch.log(zeta0)

        self.rho_by_batch = nn.Parameter(
            rho0,
            requires_grad=train_dropout,
        )

        self.log_zeta_by_batch = nn.Parameter(
            log_zeta0,
            requires_grad=train_dropout,
        )
    
    def get_dropout_params(self, batch: torch.Tensor):
        batch_idx = batch.argmax(dim=1)
        rho = self.rho_by_batch[batch_idx]
        zeta = torch.exp(self.log_zeta_by_batch[batch_idx])
        return rho, zeta

    def project(
        self,
        x: torch.Tensor,
        encode_batch: torch.Tensor,
        decode_batch: torch.Tensor,
        use_mean: bool = True,
        ) -> Dict[str, torch.Tensor]:
        """
        Project x from its original/source batch into a target/decode batch.

        Parameters
        ----------
        x:
            Input data, shape (B, n_genes).
        encode_batch:
            One-hot source batch labels, shape (B, n_batch).
            This is the original batch of each cell.
        decode_batch:
            One-hot target batch labels, shape (B, n_batch).
            This is the batch to project onto.
        use_mean:
            If True, decode z_mu for deterministic projection.
            If False, decode sampled z.
        """
        z_mu, z_logvar, z = self.encoder(x, encode_batch)

        if use_mean:
            z_dec = z_mu
        else:
            z_dec = z

        h = self.decoder_backbone(z_dec, decode_batch)
        x_mu, x_logvar = self.intensity_head(h)

        rho, zeta = self.get_dropout_params(decode_batch)

        p_miss = dropout_curve_from_mu(
            x_mu=x_mu,
            rho=rho,
            zeta=zeta,
        )

        return {
            "x_mu": x_mu,
            "x_logvar": x_logvar,
            "rho": rho,
            "zeta": zeta,
            "p_miss": p_miss,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
        }

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_mu, z_logvar, z = self.encoder(x, batch)

        h = self.decoder_backbone(z, batch)
        x_mu, x_logvar = self.intensity_head(h)
        

        rho, zeta = self.get_dropout_params(batch)

        p_miss = dropout_curve_from_mu(
            x_mu=x_mu,
            rho=rho,
            zeta=zeta,
        )

        return {
            "x_mu": x_mu,
            "x_logvar": x_logvar,
            "rho": rho,
            "zeta": zeta,
            "p_miss": p_miss,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
        }