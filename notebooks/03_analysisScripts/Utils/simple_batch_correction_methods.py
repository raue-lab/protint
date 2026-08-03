def combat_batch_correct(
    adata,
    layer_in: str | None,
    layer_out: str = "combat_corrected",
    batch_key: str = "data_source",
):
    """
    Apply ComBat batch-effect correction to a specified AnnData layer or adata.X.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data.
    layer_in : str or None, default 'threshold_controlled'
        Layer to use as input. If None, use adata.X.
    layer_out : str, default 'combat_corrected'
        Name of the layer where corrected data will be stored.
    batch_key : str, default 'data_source'
        Column in `adata.obs` containing batch identifiers.

    Returns
    -------
    AnnData
        The AnnData object with a new layer containing ComBat-corrected data.
    """
    import numpy as np
    import scanpy as sc

    # --- 0) Sanity checks ---
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")

    if layer_in is not None and layer_in not in adata.layers:
        raise KeyError(f"Input layer '{layer_in}' not found in AnnData.")

    # --- 1) Drop samples with missing batch labels ---
    mask = adata.obs[batch_key].notna()
    if mask.sum() != adata.n_obs:
        print(f"Dropping {(~mask).sum()} samples with missing '{batch_key}'")
    adata_sub = adata[mask].copy()

    # --- 2) Put data in .X for ComBat ---
    if layer_in is None:
        # Use adata.X directly
        X = adata_sub.X
    else:
        X = adata_sub.layers[layer_in]

    adata_sub.X = np.asarray(X, dtype=float)

    # --- 3) Run ComBat correction ---
    sc.pp.combat(adata_sub, key=batch_key)

    # --- 4) Store result as a new layer ---
    corrected = np.asarray(adata_sub.X, dtype=float)

    # Create output layer in full AnnData
    if layer_in is None:
        # layer_out must be explicitly stored as a layer
        adata.layers[layer_out] = np.full_like(adata.X, np.nan)
        adata.layers[layer_out][mask.values, :] = corrected
    else:
        # full layer based on layer_in
        adata.layers[layer_out] = np.full_like(adata.layers[layer_in], np.nan)
        adata.layers[layer_out][mask.values, :] = corrected

    print(f"✅ ComBat correction complete. Results stored in adata.layers['{layer_out}']")
    return adata



def limma_batch_correct(
    adata,
    layer_in: str | None ,
    layer_out: str = "limma_corrected",
    batch_key: str = "data_source",
    covariates: list | None = None,
):
    """
    Run limma::removeBatchEffect on an AnnData layer or on adata.X via rpy2.

    If `layer_in` is None, use `adata.X` as input.
    Writes corrected matrix to adata.layers[layer_out] (samples × genes).
    """
    import numpy as np
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # --- checks ---
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")

    if layer_in is not None and layer_in not in adata.layers:
        raise KeyError(f"Input layer '{layer_in}' not found in adata.layers.")

    # Drop samples with missing batch
    mask = adata.obs[batch_key].notna()
    if mask.sum() != adata.n_obs:
        print(f"Dropping {(~mask).sum()} samples with missing '{batch_key}'")
    adata_sub = adata[mask].copy()

    # --- genes × samples for limma ---
    if layer_in is None:
        X = np.asarray(adata_sub.X, dtype=float)
    else:
        X = np.asarray(adata_sub.layers[layer_in], dtype=float)

    exprs = pd.DataFrame(
        X.T,
        index=adata_sub.var_names,   # genes
        columns=adata_sub.obs_names  # samples
    )

    batch = adata_sub.obs[batch_key].astype(str).values

    # Prepare optional covariates dataframe
    cov_df = None
    if covariates:
        missing = [c for c in covariates if c not in adata_sub.obs.columns]
        if missing:
            raise KeyError(f"Covariate(s) not found in obs: {missing}")
        cov_df = adata_sub.obs[covariates].copy()
        # Ensure string columns become categories/factors in R
        for c in cov_df.columns:
            cov_df[c] = cov_df[c].astype("category")

    # --- rpy2 setup (no deprecated activate/deactivate) ---
    limma = importr("limma")

    # Send data to R (as data.frame / matrix with dimnames)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["exprs"] = ro.conversion.py2rpy(exprs)
        if cov_df is not None:
            ro.globalenv["cov_df"] = ro.conversion.py2rpy(cov_df)

    # batch as factor
    ro.globalenv["batch"] = ro.StrVector(batch)
    ro.r("batch <- factor(batch)")

    # Build design (if covariates provided) and run removeBatchEffect
    if cov_df is None:
        ro.r("exprs_corrected <- limma::removeBatchEffect(exprs, batch = batch)")
    else:
        ro.r("""
            cov_df[] <- lapply(cov_df, as.factor)
            design <- stats::model.matrix(~ 0 + ., data = cov_df)
            exprs_corrected <- limma::removeBatchEffect(exprs, batch = batch, design = design)
        """)

    # Ensure corrected matrix keeps row/col names in R
    ro.r("dimnames(exprs_corrected) <- dimnames(exprs)")

    # Retrieve as pandas DataFrame (guarantee DataFrame + labels)
    with localconverter(ro.default_converter + pandas2ri.converter):
        exprs_corr = ro.conversion.rpy2py(ro.r("exprs_corrected"))

    if not isinstance(exprs_corr, pd.DataFrame):
        exprs_corr = pd.DataFrame(
            exprs_corr,
            index=exprs.index,      # genes
            columns=exprs.columns   # samples
        )

    # Store into new layer (samples × genes), aligned to original order
    out = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=float)
    aligned = exprs_corr.T.loc[adata_sub.obs_names, adata_sub.var_names].to_numpy()
    out[mask.values, :] = aligned
    adata.layers[layer_out] = out

    print(f"✅ limma batch correction complete → adata.layers['{layer_out}']")
    return adata


def harmony_batch_correct(
    adata,
    batch_key: str = "data_source",
    layer: str | None = None,
    n_pcs: int = 50,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
):
    """
    Apply Harmony batch-effect correction on PCA embeddings.

    This function:
      1. Optionally uses a specified layer as input (otherwise uses adata.X),
      2. Runs PCA (sc.pp.pca),
      3. Runs Harmony (sc.external.pp.harmony_integrate),
      4. Stores the Harmony-corrected PCs in `adata.obsm[adjusted_basis]`.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data.

    batch_key : str, default 'data_source'
        Column in `adata.obs` containing batch identifiers.

    layer : str or None, default None
        Name of the layer to use as input for PCA.
        If None, use `adata.X` directly.

    n_pcs : int, default 50
        Number of principal components to compute before Harmony.

    basis : str, default 'X_pca'
        Key under `adata.obsm` where PCA scores are stored.

    adjusted_basis : str, default 'X_pca_harmony'
        Key under `adata.obsm` where Harmony-corrected PCs will be stored.


    Returns
    -------
    AnnData
        The same AnnData object with new PCA and Harmony embeddings in
        `adata.obsm[basis]` and `adata.obsm[adjusted_basis]`.
    """
    import numpy as np
    import scanpy as sc

    # --- 1) Drop samples with missing batch labels ---
    mask = adata.obs[batch_key].notna()
    if mask.sum() != adata.n_obs:
        print(f"Dropping {(~mask).sum()} samples with missing '{batch_key}'")

    adata_sub = adata[mask].copy()

    # --- 2) Choose data for PCA (layer or X) ---
    if layer is not None:
        adata_sub.X = np.asarray(adata_sub.layers[layer], dtype=float)

    # --- 3) PCA on chosen data ---
    sc.pp.pca(
        adata_sub,
        n_comps=n_pcs,
    )


    sc.external.pp.harmony_integrate(
        adata_sub,
        key=batch_key,
        basis=basis,
        adjusted_basis=adjusted_basis
    )

    # --- 5) Copy PCA and Harmony embeddings back to main AnnData ---
    n_obs_full = adata.n_obs

    # PCA embeddings
    pcs_sub = adata_sub.obsm[basis]
    pcs_full = np.full((n_obs_full, pcs_sub.shape[1]), np.nan, dtype=float)
    pcs_full[mask.values, :] = pcs_sub
    adata.obsm[basis] = pcs_full

    # Harmony-corrected embeddings
    harm_sub = adata_sub.obsm[adjusted_basis]
    harm_full = np.full((n_obs_full, harm_sub.shape[1]), np.nan, dtype=float)
    harm_full[mask.values, :] = harm_sub
    adata.obsm[adjusted_basis] = harm_full

    return adata