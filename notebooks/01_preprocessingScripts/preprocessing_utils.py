
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr
import matplotlib.pyplot as plt

def estimate_dropout_params_by_batch_direct(
    adata,
    batch_key="data_source",
    detected_layer="detected",
    batch_categories=None,
    robust_loss="soft_l1",
    f_scale=0.05,
):
    """
    Fit one batch-level dropout curve per batch directly in the original space:

        missing_rate(x) = 1 - Phi((x - rho) / zeta)

    where x is the per-protein median observed intensity.
    """

    def _to_dense_array(x, dtype=None):
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    batch_series = adata.obs[batch_key].astype(str)

    if batch_categories is None:
        batch_categories = sorted(batch_series.unique())
    else:
        batch_categories = [str(x) for x in batch_categories]

    X = _to_dense_array(adata.X, dtype=np.float32)
    detected = _to_dense_array(adata.layers[detected_layer], dtype=np.bool_)
    feature_names = np.asarray(adata.var.index.astype(str))

    rho_by_batch = []
    zeta_by_batch = []
    batch_rows = []
    protein_rows = []

    for batch_name in batch_categories:
        batch_mask = batch_series.to_numpy() == batch_name
        X_b = X[batch_mask]
        M_b = detected[batch_mask]

        n_detected = M_b.sum(axis=0).astype(np.int32)

        obs_rate = M_b.mean(axis=0).astype(np.float64)
        miss_rate = 1.0 - obs_rate

        masked_x = np.where(M_b, X_b, np.nan)
        medians = np.nanmedian(masked_x, axis=0).astype(np.float64)

        x_fit = medians
        y_fit = miss_rate

        def p_miss(x, rho, zeta):
            return 1.0 - ndtr((x - rho) / zeta)

        def residuals(theta):
            rho_ = theta[0]
            zeta_ = np.exp(theta[1])
            y_hat = p_miss(x_fit, rho_, zeta_)
            return  y_fit - y_hat

        idx_mid = int(np.argmin(np.abs(y_fit - 0.5)))
        rho0 = float(x_fit[idx_mid])
        zeta0 = float(np.nanstd(x_fit))

        res = least_squares(
            residuals,
            x0=np.array([rho0, np.log(zeta0)], dtype=np.float64),
            method="trf",
            loss=robust_loss,
            f_scale=f_scale,
        )

        rho = float(res.x[0])
        zeta = float(np.exp(res.x[1]))

        rho_by_batch.append(rho)
        zeta_by_batch.append(zeta)

        batch_rows.append(
            {
                "batch": batch_name,
                "rho": rho,
                "zeta": zeta,
                "n_samples": int(batch_mask.sum()),
            }
        )

        protein_rows.extend(
            {
                "batch": batch_name,
                "protein": feature_names[j],
                "median_observed_intensity": float(medians[j]),
                "observed_rate": float(obs_rate[j]),
                "missing_rate": float(miss_rate[j]),
                "missing_pct": float(miss_rate[j] * 100.0),
                "n_detected_cells": int(n_detected[j]),
            }
            for j in range(len(feature_names))
        )

    return {
        "rho_by_batch": np.asarray(rho_by_batch, dtype=np.float32),
        "zeta_by_batch": np.asarray(zeta_by_batch, dtype=np.float32),
        "params_df": pd.DataFrame(batch_rows),
        "protein_stats_df": pd.DataFrame(protein_rows),
    }

def plot_median_missingness_by_data_source_directfit(dropout_fit, figsize_per_row=(10, 4), palette = None):
    protein_df = dropout_fit["protein_stats_df"].copy()
    params_df = dropout_fit["params_df"].copy()

    batches = params_df["batch"].tolist()
    n_batches = len(batches)

    # global x-axis limits across all batches
    global_xmin = protein_df["median_observed_intensity"].min()
    global_xmax = protein_df["median_observed_intensity"].max()

    fig, axes = plt.subplots(
        n_batches,
        1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_batches),
        squeeze=False,
    )

    for i, batch_name in enumerate(batches):
        ax = axes[i, 0]

        df_b = protein_df[protein_df["batch"] == batch_name].copy()
        rho = float(params_df.loc[params_df["batch"] == batch_name, "rho"].iloc[0])
        zeta = float(params_df.loc[params_df["batch"] == batch_name, "zeta"].iloc[0])

        x = df_b["median_observed_intensity"].to_numpy(dtype=float)
        y = df_b["missing_pct"].to_numpy(dtype=float)
        
        if palette is not None:
            ax.scatter(x, y, alpha=0.5, s=20, c=palette[batch_name])
        else:
            ax.scatter(x, y, alpha=0.5, s=20)
        x_grid = np.linspace(x.min(), x.max(), 300)
        y_grid = (1.0 - ndtr((x_grid - rho) / zeta)) * 100.0

        ax.plot(
            x_grid,
            y_grid,
            color="red",
            linewidth=2.5,
            label=f"rho={rho:.2f}, zeta={zeta:.2f}",
        )
        
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_xlabel("log2 Median protein abundance")
        ax.set_ylabel("Missing prob. (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.show()