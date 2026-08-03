"""training script with cycle consistency loss."""

import os
import time
import torch
import torch.nn.functional as F
import inspect

from protint import data_utils, losses
from protint.model_mlp import MLP
from protint.init_dropout_params import initiate_dropout_params
from protint.model_utils import create_model, save_model

import argparse
import copy

import pandas as pd
import numpy as np
import numexpr
import mlflow




import scanpy as sc

#######################################################################################
# Helper functions

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def filter_kwargs_for_init(model_cls, kwargs: dict) -> dict:
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = set(params.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}

def parse_hidden_dims(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace("[", "").replace("]", "").replace(",", " ")
    parts = [p for p in s.split() if p]
    return [int(x) for x in parts]

def random_different_batch(batch_onehot: torch.Tensor) -> torch.Tensor:
    device = batch_onehot.device
    B, n_batch = batch_onehot.shape
    orig = batch_onehot.argmax(dim=1)

    if n_batch == 2:
        new = 1 - orig
    else:
        all_idx = torch.arange(n_batch, device=device).unsqueeze(0).expand(B, n_batch)
        mask = all_idx != orig.unsqueeze(1)
        candidates = all_idx[mask].view(B, n_batch - 1)
        choice = torch.randint(0, n_batch - 1, (B,), device=device)
        new = candidates[torch.arange(B, device=device), choice]

    return F.one_hot(new, num_classes=n_batch).float()

def decode_mean_with_batch(model_BatchAE, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    h = model_BatchAE.decoder_backbone(z, batch)
    x_mu, _ = model_BatchAE.intensity_head(h)
    return x_mu

def compute_cycle_loss(model_BatchAE, outputs, batch):
    """
    z --decode under switched batch--> x_mu_cycle --encode--> z_mu_cycle
    Returns: scalar minibatch sum
    """
    batch_cycle = random_different_batch(batch).to(batch.device)

    x_mu_cycle = decode_mean_with_batch(model_BatchAE, outputs["z_mu"], batch_cycle)
    enc_out = model_BatchAE.encoder(x_mu_cycle, batch_cycle)
    z_mu_cycle = enc_out[0]

    return losses.cycle_consistency_loss(outputs["z_mu"], z_mu_cycle)


#######################################################################################


def validation(model_BatchAE, val_loader, device, args, log, epoch):
    model_BatchAE.eval()

    epoch_rec_loss_val = 0.0
    epoch_kl_loss_val = 0.0
    epoch_dropout_loss_val = 0.0
    epoch_cyc_loss_val = 0.0
    epoch_tot_loss_val = 0.0

    with torch.no_grad():
        for data, detected, batch in val_loader:
            data = data.to(device)
            detected = detected.to(device)
            batch = batch.to(device)

            outputs = model_BatchAE(data, batch)

            loss_rec = losses.reconstruction_gaussian_nll(
                x_mu=outputs["x_mu"],
                x_logvar=outputs["x_logvar"],
                x=data,
                detected=detected,
            )

            loss_kl = losses.kl_loss(
                mu=outputs["z_mu"],
                logvar=outputs["z_logvar"],
            )

            loss_dropout = losses.dropout_loss(
                p_obs=outputs["p_obs"],
                detected=detected,
            )

            loss_cyc = compute_cycle_loss(model_BatchAE, outputs, batch)

            loss_total = (
                loss_rec
                + args.kl_weight * loss_kl
                + args.dropout_weight * loss_dropout
                + args.cyc_weight * loss_cyc
            )

            epoch_rec_loss_val += loss_rec.item()
            epoch_kl_loss_val += loss_kl.item()
            epoch_dropout_loss_val += loss_dropout.item()
            epoch_cyc_loss_val += loss_cyc.item()
            epoch_tot_loss_val += loss_total.item()


    log.log_metric("val_loss_rec", epoch_rec_loss_val / len(val_loader.dataset), epoch)
    log.log_metric("val_loss_kl", epoch_kl_loss_val / len(val_loader.dataset), epoch)
    log.log_metric("val_loss_dropout", epoch_dropout_loss_val / len(val_loader.dataset), epoch)
    log.log_metric("val_loss_cyc", epoch_cyc_loss_val / len(val_loader.dataset), epoch)
    log.log_metric("val_loss_tot", epoch_tot_loss_val / len(val_loader.dataset), epoch)

    return epoch_tot_loss_val / len(val_loader.dataset)


def train_model(
    model_BatchAE,
    optimizer_BatchAE,
    train_loader,
    val_loader,
    run_dir,
    device,
    log,
    args,
    model_src_adv=None,
    optimizer_src_adv=None,
    src_weights_src_adv=None,
):
    best_model_loss = np.inf
    waited_epochs = 0
    early_stop = False

    ae_model_file = os.path.join(run_dir, "models", "batch_ae_final.model")
    src_model_file = os.path.join(run_dir, "models", "src_adv_final.model")

    for epoch in range(args.epochs):
        if early_stop:
            break

        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_dropout_loss = 0.0
        epoch_cyc_loss = 0.0
        epoch_src_adv_loss = 0.0
        epoch_tot_loss = 0.0

        model_BatchAE.train()
        if (args.src_adv_weight > 0):
            model_src_adv.train()

        for data, detected, batch in train_loader:
            data = data.to(device)
            detected = detected.to(device)
            batch = batch.to(device)

            # ------------------------------------------------------------
            # Forward
            # ------------------------------------------------------------
            outputs = model_BatchAE(data, batch)

            # ------------------------------------------------------------
            # Adversary update step
            # ------------------------------------------------------------
            if (args.src_adv_weight > 0):
                set_requires_grad(model_src_adv, True)
                optimizer_src_adv.zero_grad(set_to_none=True)

                src_pred = model_src_adv(outputs["z"].detach())
                loss_src_adv = losses.adversarial_loss(src_pred, batch, src_weights_src_adv)
                loss_src_adv.backward()
                optimizer_src_adv.step()

                epoch_src_adv_loss += loss_src_adv.item()

                # adversarial penalty back to encoder
                set_requires_grad(model_src_adv, False)
                src_pred_for_encoder = model_src_adv(outputs["z"])
                loss_src_adv_for_encoder = losses.adversarial_loss(src_pred_for_encoder, batch, src_weights_src_adv)
            else:
                loss_src_adv_for_encoder = torch.tensor(0.0, device=device)

            # ------------------------------------------------------------
            # Main model update
            # ------------------------------------------------------------
            optimizer_BatchAE.zero_grad(set_to_none=True)

            loss_rec = losses.reconstruction_gaussian_nll(
                x_mu=outputs["x_mu"],
                x_logvar=outputs["x_logvar"],
                x=data,
                detected=detected,
            )

            loss_kl = losses.kl_loss(
                mu=outputs["z_mu"],
                logvar=outputs["z_logvar"],
            )

            loss_dropout = losses.dropout_loss(
                p_miss=outputs["p_miss"],
                detected=detected,
            )

            loss_cyc = compute_cycle_loss(model_BatchAE, outputs, batch)

            loss_total = (
                loss_rec
                + args.kl_weight * loss_kl
                + args.dropout_weight * loss_dropout
                + args.cyc_weight * loss_cyc
                - args.src_adv_weight * loss_src_adv_for_encoder
            )

            loss_total.backward()
            optimizer_BatchAE.step()

            if (args.src_adv_weight > 0):
                set_requires_grad(model_src_adv, True)

            epoch_rec_loss += loss_rec.detach().item()
            epoch_kl_loss += loss_kl.detach().item()
            epoch_dropout_loss += loss_dropout.detach().item()
            epoch_cyc_loss += loss_cyc.detach().item()
            epoch_tot_loss += loss_total.detach().item()


        log.log_metric("train_loss_rec", epoch_rec_loss / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_kl", epoch_kl_loss / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_dropout", epoch_dropout_loss / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_cyc", epoch_cyc_loss / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_adv", epoch_src_adv_loss / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_tot", epoch_tot_loss / len(train_loader.dataset), epoch)

        if args.val_set_size != 0:
            epoch_loss_val = validation(model_BatchAE, val_loader, device, args, log, epoch)

            if epoch_loss_val < best_model_loss:
                best_model_loss = epoch_loss_val
                waited_epochs = 0

                save_model(
                    model_BatchAE,
                    optimizer_BatchAE,
                    epoch,
                    epoch_tot_loss / len(train_loader.dataset),
                    ae_model_file,
                    device,
                )

                if (model_src_adv is not None) and (optimizer_src_adv is not None) and (args.src_adv_weight > 0):
                    save_model(
                        model_src_adv,
                        optimizer_src_adv,
                        epoch,
                        epoch_src_adv_loss / len(train_loader.dataset),
                        src_model_file,
                        device,
                    )
            else:
                waited_epochs += 1
                if waited_epochs > args.patience:
                    early_stop = True

    if args.val_set_size == 0:

        save_model(
            model_BatchAE,
            optimizer_BatchAE,
            epoch,
            epoch_tot_loss / len(train_loader.dataset),
            ae_model_file,
            device,
        )

        if (model_src_adv is not None) and (optimizer_src_adv is not None) and (args.src_adv_weight > 0):
            save_model(
                model_src_adv,
                optimizer_src_adv,
                epoch,
                epoch_src_adv_loss / len(train_loader.dataset),
                src_model_file,
                device,
            )


def main(args, model_cls):
    if args.use_mlflow:
        run_dir = os.path.join(args.tmp_dir, str(int(time.time())))
        mlflow.set_tracking_uri(args.mlflow_storage_path)
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)
    else: 
        run_dir = args.output_dir
    
    data_utils.create_temp_dirs(run_dir)
    
    log = data_utils.log_obj(args.use_mlflow, run_dir)
    log.log_params(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    numexpr.set_num_threads(numexpr.detect_number_of_cores())
    
    adata = sc.read(args.train_file)
    
    train_loader, val_loader, label_encode = data_utils.create_dataloaders_from_adata(
        adata, 
        args.batch_size, 
        args.val_set_size, 
        args.random_seed,
        args.use_sparse_mat
    )
    dropout_params_df = initiate_dropout_params(
        adata,
        batch_key="data_source",
        detected_layer="detected"
    )

    # Save features and label encoding
    features = adata.var.index.to_frame()
    label_encode.to_csv(os.path.join(run_dir, 'models', 'label_encode.csv'))
    features.to_csv(os.path.join(run_dir, 'models', 'features.csv'))
    
    set_seed(args.random_seed)
    
    # ---- Build model kwargs from args (may include new stuff) ----
    model_kwargs = {}

    hd = parse_hidden_dims(getattr(args, "hidden_dims", None))
    if hd:
        model_kwargs["hidden_dims"] = tuple(hd)

    # kwargs for weight of the losses
    model_kwargs["kl_weight"] = args.kl_weight
    model_kwargs["cyc_weight"] = args.cyc_weight
    model_kwargs["dropout_weight"] = args.dropout_weight
    model_kwargs["src_adv_weight"] = args.src_adv_weight

    # activation/norm/etc 
    model_kwargs["activation"] = args.activation
    model_kwargs["norm"] = args.norm
    model_kwargs["norm_eps"] = args.norm_eps
    model_kwargs["bn_momentum"] = args.bn_momentum
    model_kwargs["gn_groups"] = args.gn_groups
    model_kwargs["dropout_p"] = args.dropout_p
    model_kwargs["dropout_params_df"] = dropout_params_df
    model_kwargs["train_dropout"] = args.train_dropout

    # ---- Filter to what this model actually accepts (old/new safe) ----
    model_kwargs = filter_kwargs_for_init(model_cls, model_kwargs)

    # ---- Create model ----
    ## Create the batch VAE model 
    model_BatchAE, optimizer_BatchAE = create_model(
        model_cls,
        device,
        features.shape[0],
        args.encoding_dim,
        label_encode.shape[0],
        lr=args.batch_lr,
        filename=None,
        **model_kwargs,
    )


    ## create the adversarial neural network
    if args.src_adv_weight > 0:
        model_src_adv = MLP(
            enc_dim=args.encoding_dim,
            output_dim=label_encode.shape[0],
            activation=getattr(args, "activation", "softplus"),
            norm=getattr(args, "norm", "layernorm"),
            norm_eps=getattr(args, "norm_eps", 1e-5),
            bn_momentum=getattr(args, "bn_momentum", 0.1),
            gn_groups=getattr(args, "gn_groups", 8),
        ).to(device)

        optimizer_src_adv = torch.optim.Adam(model_src_adv.parameters(), lr=args.src_adv_lr)

        src_weights_src_adv = torch.tensor(data_utils.get_class_weights(adata.obs.data_source, args.balanced_sources_src_adv), dtype=torch.float).to(device)
    else:
        model_src_adv = None
        optimizer_src_adv = None
        src_weights_src_adv = None
    
    train_model(
        model_BatchAE, 
        optimizer_BatchAE,
        train_loader, 
        val_loader, 
        run_dir,
        device,
        log,
        args,
        model_src_adv=model_src_adv,
        optimizer_src_adv=optimizer_src_adv,
        src_weights_src_adv=src_weights_src_adv,
    )
    # Save hidden_dims explicitly (string) if used
    params_path = os.path.join(run_dir, "models", "params.csv")
    df = pd.read_csv(params_path, index_col=0)

    hd = getattr(args, "hidden_dims", None)

    if hd is not None:
        if isinstance(hd, (list, tuple)):
            hidden_str = " ".join(map(str, hd))
        else:
            hidden_str = str(hd).replace("[", "").replace("]", "").replace(",", " ")

        hidden_str = " ".join(hidden_str.split())

        if "hidden_dims" not in df.index:
            df.loc["hidden_dims"] = np.nan

        df.loc["hidden_dims", "value"] = hidden_str

    df.to_csv(params_path)

    dropout_params_df.to_csv(
        os.path.join(run_dir, "models", "dropout_curve_by_batch.csv"),
        index=False,
    )

    log.end_log()