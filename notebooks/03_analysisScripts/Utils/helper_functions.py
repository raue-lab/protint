
def load_loss_file(path, label):
    import pandas as pd
    import os
    if not os.path.exists(path):
        print(f"⚠️ Warning: {label} file not found: {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    if "epoch" not in df or "value" not in df:
        raise ValueError(f"File {path} requires 'epoch' & 'value' columns.")
    return df

def get_last_line(path):
    """Return the last non-empty line of a text file, or None if file missing/empty."""
    import os

    if not os.path.isfile(path):
        return None
    last = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return last

def add_joint_annotations(z_adata, manual_mapping):
    """
    Add joint_Tissue type, joint_Cancer type, joint_Cancer subtype to z_adata.obs
    using the preloaded tissue_map, ctype_map, subtype_map.
    """
    obs = z_adata.obs.copy()
    obs["cell_id"] = obs.index

    obs = obs.merge(
        manual_mapping,
        how="left",
        on=["data_source", "Tissue type", "Cancer type", "Cancer subtype"]
    )
    obs = obs.set_index("cell_id")
    z_adata.obs = obs

    return z_adata

def subset_by_flag(z_adata, flag_col):
    """
    Return a copy of z_adata subset to rows where obs[flag_col] is True.
    If column missing or subset empty, return None.
    """
    if flag_col not in z_adata.obs.columns:
        print(f"[WARN] Flag column '{flag_col}' not found in z_adata.obs", flush=True)
        return None

    mask = z_adata.obs[flag_col].fillna(False).astype(bool).values
    if mask.sum() == 0:
        print(f"[WARN] Subset for '{flag_col}' is empty", flush=True)
        return None

    return z_adata[mask].copy()


def run_lisi_at_k(
    z_adata,
    ks=[15, 30, 60],
    batch_key="data_source",
    prefix="all",
    n_pcs=50,
    use_rep=None,
):
    """
    Compute iLISI and cLISI scores for multiple neighborhood sizes k.

    Behavior
    --------
    If use_rep is None:
        PCA is computed from z_adata.X with n_comps=n_pcs, and LISI is
        computed from z_adata.obsm['X_pca'].

    If use_rep is not None:
        LISI is computed from the specified representation, e.g.
        'X_pca', 'X_pca_harmony', or 'X_pca_harmony_theta2'.
    """
    import scanpy as sc
    import scib

    metrics = {}

    if use_rep is None:
        if n_pcs is None:
            raise ValueError(
                "When use_rep=None, please provide n_pcs, e.g. n_pcs=50."
            )

        print(
            f"[LISI:{prefix}] use_rep=None, computing PCA with n_pcs={n_pcs}",
            flush=True,
        )

        sc.pp.pca(
            z_adata,
            n_comps=n_pcs,
        )

        lisi_use_rep = "X_pca"
        lisi_n_pcs = n_pcs

    else:
        lisi_use_rep = use_rep
        lisi_n_pcs = n_pcs

    for k in ks:
        print(
            f"[LISI:{prefix}] Running neighbors with k={k}, "
            f"use_rep={lisi_use_rep}, n_pcs={lisi_n_pcs}",
            flush=True,
        )

        sc.pp.neighbors(
            z_adata,
            n_neighbors=k,
            use_rep=lisi_use_rep,
            n_pcs=lisi_n_pcs,
            metric="euclidean",
        )

        ilisi = scib.me.ilisi_graph(
            z_adata,
            batch_key=batch_key,
            type_="knn",
        )

        clisi_tissue = scib.me.clisi_graph(
            z_adata,
            label_key="joint tissue type",
            type_="knn",
        )

        clisi_ctype = scib.me.clisi_graph(
            z_adata,
            label_key="joint Cancer type",
            type_="knn",
        )

        clisi_csub = scib.me.clisi_graph(
            z_adata,
            label_key="joint Cancer subtype",
            type_="knn",
        )

        metrics[f"{prefix}_k{k}_iLISI"] = float(ilisi)
        metrics[f"{prefix}_k{k}_cLISI_tissue"] = float(clisi_tissue)
        metrics[f"{prefix}_k{k}_cLISI_ctype"] = float(clisi_ctype)
        metrics[f"{prefix}_k{k}_cLISI_csubtype"] = float(clisi_csub)

    return metrics

def get_batch_vector(label_encode, batch_name, expected_dim=None):
    """
    Return one-hot vector for batch_name.

    Supports both conventions:

    1. Preferred:
        label_encode.index = batch names
        label_encode.columns = encoded dimensions

    2. Older:
        label_encode.columns = batch names
        label_encode.index = encoded dimensions

    If expected_dim is provided, the returned vector must match it.
    """
    import numpy as np

    batch_name = str(batch_name)

    le = label_encode.copy()
    le.index = le.index.astype(str)
    le.columns = le.columns.astype(str)

    candidates = []

    if batch_name in le.index:
        candidates.append(le.loc[batch_name].to_numpy(dtype=np.float32))

    if batch_name in le.columns:
        candidates.append(le[batch_name].to_numpy(dtype=np.float32))

    if not candidates:
        raise KeyError(
            f"Batch {batch_name!r} not found in label_encode index or columns. "
            f"Index examples: {list(le.index)[:10]}; "
            f"column examples: {list(le.columns)[:10]}"
        )

    if expected_dim is not None:
        candidates = [v for v in candidates if v.shape[0] == expected_dim]

        if not candidates:
            raise ValueError(
                f"Batch {batch_name!r} was found in label_encode, but no encoding "
                f"vector had expected length {expected_dim}."
            )

    return candidates[0]


def get_batch_matrix(label_encode, batch_names, expected_dim=None):
    """
    Convert batch labels into a one-hot matrix.
    """
    import numpy as np

    return np.stack(
        [
            get_batch_vector(
                label_encode=label_encode,
                batch_name=b,
                expected_dim=expected_dim,
            )
            for b in batch_names
        ],
        axis=0,
    ).astype(np.float32)



def reconstruct_with_original_batch(
    model,
    adata,
    label_encode,
    device,
    decode_matrix,          # import from protint.projection
    batch_key="data_source",
    decimals=4,
    batch_size=1600,
    use_sparse_mat=False,
    use_mean=True,
):
    """
    Reconstruct adata using each sample's original batch label.

    Correct reconstruction logic:
        z_mu = encoder(x, original_batch)
        x_recon = decoder(z_mu, original_batch)

    This mirrors projection, but with:
        encode_batch == decode_batch == original_batch
    """

    import torch
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import csr_matrix
    from torch.utils.data import DataLoader, TensorDataset

    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"Required batch column {batch_key!r} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Infer expected one-hot dimension from the model.
    expected_dim = model.encoder.batch_fc.in_features

    # Original expression matrix.
    X = decode_matrix(
        adata,
        use_sparse_mat=use_sparse_mat,
    )

    # Original batch labels per cell.
    batches = adata.obs[batch_key].astype(str).to_numpy()

    # One batch-encoding vector per cell.
    batch_label = get_batch_matrix(
        label_encode=label_encode,
        batch_names=batches,
        expected_dim=expected_dim,
    )

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(batch_label, dtype=torch.float32),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    reconstructed = []
    encoded = []

    model.eval()

    with torch.no_grad():
        for data, batch in data_loader:
            data = data.to(device)
            batch = batch.to(device)

            outputs = model.project(
                x=data,
                encode_batch=batch,
                decode_batch=batch,
                use_mean=use_mean,
            )

            reconstructed.append(outputs["x_mu"])
            encoded.append(outputs["z_mu"])

    reconstructed = (
        torch.cat(reconstructed, dim=0)
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

    if use_sparse_mat:
        recon_X = csr_matrix(reconstructed)
    else:
        recon_X = reconstructed

    recon_adata = sc.AnnData(
        recon_X,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    z_adata = sc.AnnData(
        encoded,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=[f"z_{i}" for i in range(encoded.shape[1])]),
    )

    recon_adata.obs["reconstructed_with"] = batches

    return recon_adata, z_adata