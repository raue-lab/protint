import os
import inspect

import torch
import numpy as np
import pandas as pd
import scanpy as sc

from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset

from protint.model_utils import create_model


def filter_kwargs_for_init(model_cls, kwargs: dict) -> dict:
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = set(params.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}

def get_batch_vector(label_encode: pd.DataFrame, batch_name) -> np.ndarray:
    """
    Return one-hot vector for a batch.

    Preferred format:
        label_encode.index = batch names
        label_encode.columns = one-hot dimensions

    Fallback:
        label_encode.columns = batch names
    """
    if batch_name in label_encode.index:
        return label_encode.loc[batch_name].to_numpy(dtype=np.float32)

    if batch_name in label_encode.columns:
        return label_encode[batch_name].to_numpy(dtype=np.float32)

    raise KeyError(
        f"Batch {batch_name!r} not found in label_encode index or columns. "
        f"Available index values: {list(label_encode.index)[:10]}; "
        f"available columns: {list(label_encode.columns)[:10]}"
    )


def get_batch_matrix(label_encode: pd.DataFrame, batch_names) -> np.ndarray:
    """
    Convert a sequence of batch labels into a one-hot matrix.
    """
    return np.stack(
        [get_batch_vector(label_encode, b) for b in batch_names],
        axis=0,
    ).astype(np.float32)

def decode_projection(data_loader, model, device, decimals, use_mean=True):
    decoded = []
    encoded = []

    model.eval()

    with torch.no_grad():
        for data, encode_batch, decode_batch in data_loader:
            data = data.to(device)
            encode_batch = encode_batch.to(device)
            decode_batch = decode_batch.to(device)

            outputs = model.project(
                x=data,
                encode_batch=encode_batch,
                decode_batch=decode_batch,
                use_mean=use_mean,
            )

            decoded.append(outputs["x_mu"])
            encoded.append(outputs["z_mu"])

    decoded = (
        torch.cat(decoded, dim=0)
        .detach()
        .cpu()
        .numpy()
        .round(decimals=decimals)
    )

    encoded = (
        torch.cat(encoded, dim=0)
        .detach()
        .cpu()
        .numpy()
        .round(decimals=decimals)
    )

    return decoded, encoded


def load_model(model_dir, device, model_cls):
    features = pd.read_csv(os.path.join(model_dir, "features.csv"), index_col=0).index
    label_encode = pd.read_csv(os.path.join(model_dir, "label_encode.csv"), index_col=0)
    params = pd.read_csv(os.path.join(model_dir, "params.csv"), index_col=0)
    dropout_params_df = pd.read_csv(os.path.join(model_dir, "dropout_curve_by_batch.csv"))

    n_genes = features.shape[0]
    n_batch = label_encode.shape[0]
    enc_dim = int(float(params.loc["encoding_dim", "value"]))

    hidden_dims = str(params.loc["hidden_dims", "value"])
    hidden_dims = hidden_dims.replace("[", "").replace("]", "").replace(",", " ")
    hidden_dims = tuple(int(x) for x in hidden_dims.split())

    train_dropout = (
        str(params.loc["train_dropout", "value"])
        .strip()
        .lower()
        in {"true", "1", "yes", "y"}
    )

    model_kwargs = {
        "hidden_dims": hidden_dims,
        "kl_weight": float(params.loc["kl_weight", "value"]),
        "cyc_weight": float(params.loc["cyc_weight", "value"]),
        "dropout_weight": float(params.loc["dropout_weight", "value"]),
        "src_adv_weight": float(params.loc["src_adv_weight", "value"]),
        "activation": str(params.loc["activation", "value"]),
        "norm": str(params.loc["norm", "value"]),
        "norm_eps": float(params.loc["norm_eps", "value"]),
        "bn_momentum": float(params.loc["bn_momentum", "value"]),
        "gn_groups": int(float(params.loc["gn_groups", "value"])),
        "dropout_p": float(params.loc["dropout_p", "value"]),
        "dropout_params_df": dropout_params_df,
        "train_dropout": train_dropout,
    }

    model_kwargs = filter_kwargs_for_init(model_cls, model_kwargs)

    ckpt_path = os.path.join(model_dir, "batch_ae_final.model")

    model, _ = create_model(
        model_cls,
        device,
        n_genes,
        enc_dim,
        n_batch,
        lr=float(params.loc["batch_lr", "value"]),
        filename=ckpt_path,
        **model_kwargs,
    )

    model.eval()

    return model, features, label_encode


def decode_matrix(adata, use_sparse_mat=False):
    if use_sparse_mat:
        return adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    try:
        X = adata.X.todense()
    except Exception:
        X = adata.X

    return np.asarray(X)


def do_projection(
    model,
    adata,
    onto,
    label_encode,
    device,
    decimals=4,
    batch_size=1600,
    use_sparse_mat=False,
    use_mean=True,
):
    """
    Project cells from their original `data_source` batch into the target batch `onto`.

    Correct projection logic:
        z_mu = encoder(x, source_batch)
        x_projected = decoder(z_mu, target_batch)
    """

    source_batch_names = adata.obs["data_source"].to_numpy()

    encode_label = get_batch_matrix(
        label_encode=label_encode,
        batch_names=source_batch_names,
    )

    target_vec = get_batch_vector(
        label_encode=label_encode,
        batch_name=onto,
    )

    decode_label = np.repeat(
        target_vec.reshape(1, -1),
        repeats=adata.shape[0],
        axis=0,
    ).astype(np.float32)

    X = decode_matrix(adata, use_sparse_mat=use_sparse_mat)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(encode_label, dtype=torch.float32),
        torch.tensor(decode_label, dtype=torch.float32),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    projected, z = decode_projection(
        data_loader=data_loader,
        model=model,
        device=device,
        decimals=decimals,
        use_mean=use_mean,
    )

    if use_sparse_mat:
        proj_adata = sc.AnnData(
            csr_matrix(projected),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
    else:
        proj_adata = sc.AnnData(
            projected,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )

    proj_adata.obs["projected_onto"] = onto
    proj_adata.obs["projection_source_batch"] = source_batch_names

    z_adata = sc.AnnData(
        z,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=[f"z_{i}" for i in range(z.shape[1])]),
    )

    return proj_adata, z_adata


def main(args, model_cls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read(args.projection_file)

    model, features, label_encode = load_model(
        args.model_dir,
        device,
        model_cls,
    )

    adata = adata[:, features].copy()

    proj_adata, z_adata = do_projection(
        model=model,
        adata=adata,
        onto=args.onto,
        label_encode=label_encode,
        device=device,
        decimals=args.decimals,
        batch_size=getattr(args, "batch_size", 1600),
        use_sparse_mat=getattr(args, "use_sparse_mat", False),
        use_mean=getattr(args, "use_mean", True),
    )

    proj_adata.write(args.output_file)

    if hasattr(args, "z_output_file") and args.z_output_file is not None:
        z_adata.write(args.z_output_file)