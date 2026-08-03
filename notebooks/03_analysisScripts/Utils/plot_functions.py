# Collection of plotting functions related to MOBER and its output
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

def plot_training_loss(
    model_output_dir,
    *,
    adv_file="train_loss_adv",
    ae_file="train_loss_ae",
    cyc_file="train_loss_cyc",
    metrics_dir="metrics",
    figsize=(7, 4),
    title="Training Loss (log scale)",
    hline=np.log(2),                # reference horizontal line
    palette=None,                  # <-- dict mapping loss key → color
    save=None
):
    """
    Plot ProtInt training losses over epochs (adversarial, autoencoder, cycle consistency)
    with optional custom color mapping for each loss curve.
    
    palette: dict with optional keys:
        {
            "adversarial": "blue",
            "autoencoder": "green",
            "cycle": "orange"
        }
    """
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Paths
    adv_path = os.path.join(model_output_dir, metrics_dir, adv_file)
    ae_path = os.path.join(model_output_dir, metrics_dir, ae_file)
    cyc_path = os.path.join(model_output_dir, metrics_dir, cyc_file)
    dropout_path = os.path.join(model_output_dir, metrics_dir, "train_loss_dropout")
    # Load helper
    def load_loss_file(path, label):
        if not os.path.exists(path):
            print(f"⚠️ Warning: {label} file not found: {path}")
            return None
        df = pd.read_csv(path, sep="\t")
        if "epoch" not in df or "value" not in df:
            raise ValueError(f"File {path} requires 'epoch' & 'value' columns.")
        return df

    training_adv = load_loss_file(adv_path, "Adversarial")
    training_ae = load_loss_file(ae_path, "Autoencoder")
    training_cyc = load_loss_file(cyc_path, "Cycle Consistency")
    training_dropout = load_loss_file(dropout_path, "Dropout")
    
    # Helper for choosing colors
    def get_color(key):
        if isinstance(palette, dict) and key in palette:
            return palette[key]
        return None  # fallback to seaborn default

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if training_adv is not None:
        sns.lineplot(
            data=training_adv, x="epoch", y="value",
            label="adversarial", linewidth=2, ax=ax,
            color=get_color("adversarial")
        )

    if training_ae is not None:
        sns.lineplot(
            data=training_ae, x="epoch", y="value",
            label="autoencoder", linewidth=2, ax=ax,
            color=get_color("autoencoder")
        )

    if training_cyc is not None:
        sns.lineplot(
            data=training_cyc, x="epoch", y="value",
            label="cycle consistency", linewidth=2, ax=ax,
            color=get_color("cycle")
        )

    if training_dropout is not None:
        sns.lineplot(
            data=training_dropout, x="epoch", y="value",
            label="dropout", linewidth=2, ax=ax,
            color=get_color("dropout")
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if title:
        ax.set_title(title)

    # Horizontal reference line (always red)
    if hline is not None:
        ax.axhline(
            hline, linestyle="--", linewidth=1.6,
            color="red", alpha=0.8,
            label=f"reference = {round(hline,3)}"
        )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")

    return fig, ax



#######################################################################################################
def plot_validation_loss(
    model_output_dir,
    *,
    adv_file="val_loss_adv",
    ae_file="val_loss_ae",
    cyc_file="val_loss_cyc",
    metrics_dir="metrics",
    figsize=(7, 4),
    title="Validation Loss (log scale)",
    hline=np.log(2),     # reference baseline (optional)
    save=None
):
    """
    Plot ProtInt validation losses over epochs (adversarial + autoencoder),
    including an optional horizontal reference line (e.g., ln(2)).
    """
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Paths
    adv_path = os.path.join(model_output_dir, metrics_dir, adv_file)
    ae_path = os.path.join(model_output_dir, metrics_dir, ae_file)
    cyc_path = os.path.join(model_output_dir, metrics_dir, cyc_file)

    # Load helper
    def load_loss_file(path, label):
        if not os.path.exists(path):
            print(f"⚠️ Warning: {label} file not found: {path}")
            return None
        df = pd.read_csv(path, sep="\t")
        if "epoch" not in df or "value" not in df:
            raise ValueError(f"File {path} requires 'epoch' & 'value' columns.")
        return df

    val_adv = load_loss_file(adv_path, "Validation Adversarial")
    val_ae  = load_loss_file(ae_path, "Validation Autoencoder")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if val_adv is not None:
        sns.lineplot(val_adv, x="epoch", y="value",
                     label="val adversarial", linewidth=2, ax=ax)
    if val_ae is not None:
        sns.lineplot(val_ae, x="epoch", y="value",
                     label="val autoencoder", linewidth=2, ax=ax)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if title:
        ax.set_title(title)

    # Optional horizontal reference line
    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.6,
                   color="red", alpha=0.8,
                   label=f"reference = {round(hline,3)}")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")

    return fig, ax

##############################################################################################################################
# comparing recon_adata with adata, by joint UMAP
def joint_umap_original_reconstructed(
    adata,
    recon_adata,
    detected_layer="detected",
    n_comps=50,
    n_pcs=50,
    scale=True,
    max_value=10,
    size=20,
    line_width=0.4,
    line_alpha=1,
    line_color="black",
    point_alpha=0.8,
    figsize=(6,3.75),
    random_state=0,
    palette=("tab:green", "tab:purple"),
    return_joint=True,
):
    """
    Run joint PCA + UMAP on original and reconstructed data, using only originally
    observed values, and draw lines connecting matched original/reconstructed cells.

    Parameters
    ----------
    adata
        Original AnnData. Must contain `adata.layers[detected_layer]`.
    recon_adata
        Reconstructed AnnData with the same cells and variables as `adata`.
    detected_layer
        Layer in `adata` containing True/False detection mask.
    n_comps
        Number of PCA components.
    n_pcs
        Number of PCs used for neighbors/UMAP.
    scale
        Whether to run `sc.pp.scale`.
    max_value
        Passed to `sc.pp.scale`.
    size
        UMAP point size.
    line_width
        Width of lines connecting original/reconstructed cells.
    line_alpha
        Transparency of connecting lines.
    point_alpha
        Transparency of UMAP points.
    figsize
        Figure size.
    random_state
        Random seed for UMAP.
    return_joint
        If True, return the joint AnnData and matplotlib objects.

    Returns
    -------
    joint, fig, ax
        If `return_joint=True`.

    Otherwise returns None.
    """

    def as_dense(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)

    if detected_layer not in adata.layers:
        raise KeyError(f"`adata.layers[{detected_layer!r}]` was not found.")

    if adata.shape != recon_adata.shape:
        raise ValueError(
            f"`adata` and `recon_adata` must have the same shape. "
            f"Got {adata.shape} and {recon_adata.shape}."
        )

    if not np.array_equal(adata.var_names, recon_adata.var_names):
        raise ValueError("`adata.var_names` and `recon_adata.var_names` must match.")

    if not np.array_equal(adata.obs_names, recon_adata.obs_names):
        raise ValueError("`adata.obs_names` and `recon_adata.obs_names` must match.")

    X_orig = as_dense(adata.X)
    X_recon = as_dense(recon_adata.X)
    detected = as_dense(adata.layers[detected_layer]).astype(bool)

    if X_orig.shape != detected.shape:
        raise ValueError(
            f"`adata.X` and `adata.layers[{detected_layer!r}]` must have the same shape. "
            f"Got {X_orig.shape} and {detected.shape}."
        )

    # Keep only originally observed values in both original and reconstructed matrices.
    X_orig_observed_only = X_orig.copy()
    X_recon_observed_only = X_recon.copy()

    X_orig_observed_only[~detected] = 0
    X_recon_observed_only[~detected] = 0

    orig_joint = sc.AnnData(
        X_orig_observed_only,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    recon_joint = sc.AnnData(
        X_recon_observed_only,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    orig_joint.obs["data_type"] = "original"
    recon_joint.obs["data_type"] = "reconstructed"

    orig_joint.obs["pair_id"] = adata.obs_names.astype(str)
    recon_joint.obs["pair_id"] = adata.obs_names.astype(str)

    joint = ad.concat(
        [orig_joint, recon_joint],
        axis=0,
        keys=["original", "reconstructed"],
        index_unique="-",
    )

    joint.obs["data_type"] = joint.obs["data_type"].astype(str)
    joint.obs["pair_id"] = joint.obs["pair_id"].astype(str)

    if scale:
        sc.pp.scale(joint, max_value=max_value)

    max_possible_pcs = min(joint.n_obs - 1, joint.n_vars)
    actual_n_comps = min(n_comps, max_possible_pcs)
    actual_n_pcs = min(n_pcs, actual_n_comps)

    sc.tl.pca(joint, n_comps=actual_n_comps)
    sc.pp.neighbors(joint, n_pcs=actual_n_pcs)
    sc.tl.umap(joint, random_state=random_state)

    fig, ax = plt.subplots(figsize=figsize)

    sc.pl.umap(
        joint,
        color="data_type",
        ax=ax,
        show=False,
        size=size,
        alpha=point_alpha,
        palette=palette,
    )

    umap = joint.obsm["X_umap"]

    obs = joint.obs.copy()
    obs["_umap_x"] = umap[:, 0]
    obs["_umap_y"] = umap[:, 1]

    orig = obs[obs["data_type"] == "original"].set_index("pair_id")
    recon = obs[obs["data_type"] == "reconstructed"].set_index("pair_id")

    common_ids = sorted(set(orig.index) & set(recon.index))

    for cell_id in common_ids:
        x1, y1 = orig.loc[cell_id, ["_umap_x", "_umap_y"]]
        x2, y2 = recon.loc[cell_id, ["_umap_x", "_umap_y"]]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=line_width,
            alpha=line_alpha,
            color=line_color,
            zorder=10,
        )

    ax.set_title("Joint UMAP: original vs reconstructed")
    plt.tight_layout()
    plt.show()

    if return_joint:
        return joint, fig, ax




def plot_detected_reconstructed_imputed_histogram(
    adata,
    recon_adata,
    bins: int = 80,
    density: bool = True,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (5, 4),
):
    """
    Plot intensity distributions for:
      1. Input intensities from adata.X where adata.layers['detected'] == True
      2. Reconstructed intensities from recon_adata.X where adata.layers['detected'] == True
      3. Imputed intensities from recon_adata.X where adata.layers['detected'] == False
    """

    def _to_numpy(x):
        if sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x)

    adata_intensity = _to_numpy(adata.X)
    recon_intensity = _to_numpy(recon_adata.X)

    detected = _to_numpy(adata.layers["detected"]).astype(bool)

    input_intensities = adata_intensity[detected]
    reconstructed_detected_intensities = recon_intensity[detected]
    imputed_intensities = recon_intensity[~detected]

    plt.figure(figsize=figsize)

    plt.hist(
        input_intensities,
        bins=bins,
        alpha=alpha,
        density=density,
        label=f"Input",
        color="green"
    )

    plt.hist(
        reconstructed_detected_intensities,
        bins=bins,
        alpha=alpha,
        density=density,
        label=f"Reconstructed",
        color="purple"
    )

    plt.hist(
        imputed_intensities,
        bins=bins,
        alpha=alpha,
        density=density,
        label=f"Imputed",
        color="orange"
    )

    plt.xlabel("Intensity")
    plt.ylabel("Density" if density else "Count")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.show()