import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import least_squares
from scipy.special import ndtr


def get_batch_categories_from_label_encode(label_encode):
    if isinstance(label_encode, pd.Series):
        return label_encode.sort_values().index.astype(str).tolist()

    if isinstance(label_encode, pd.DataFrame):
        if label_encode.shape[1] == 1:
            return label_encode.iloc[:, 0].sort_values().index.astype(str).tolist()

    raise ValueError("Could not infer batch category order from label_encode")

def initiate_dropout_params(
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

    where:
        - x is the per-protein median observed intensity within that batch
        - Phi is the standard normal CDF
        - rho is the intensity at which the fitted missing rate is about 0.5
        - zeta controls the steepness of the transition

    The fit is performed directly in probability space by nonlinear least squares,
    using a robust loss through `scipy.optimize.least_squares`. The parameter
    `zeta` is optimized on the log scale to enforce positivity.
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

    batch_rows = []

    for batch_name in batch_categories:
        batch_mask = batch_series.to_numpy() == batch_name
        X_b = X[batch_mask]
        M_b = detected[batch_mask]


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

        batch_rows.append(
            {
                "batch": batch_name,
                "rho": rho,
                "zeta": zeta,
                "n_samples": int(batch_mask.sum()),
            }
        )


    return  pd.DataFrame(batch_rows)
