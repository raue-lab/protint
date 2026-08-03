import anndata as ad
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns



# Setup logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def proteome_2_anndata(proteome_df: pd.DataFrame, 
                       metadata_df: pd.DataFrame, 
                       match_sample_column: str, 
                       match_protein_column: str, 
                       protein_annotation_columns: list, 
                       need_log2: bool = True,
                       log2_intensity_threshold: float = None):
    """
    Build an :class:`anndata.AnnData` object from a wide proteomics matrix and sample metadata.

    This routine:
      1) Aligns samples between `proteome_df` and `metadata_df`.
      2) Keeps only protein-annotation columns + matched sample columns (drops others with a warning).
      3) Cleans sample columns (',' → '.', numeric coercion).
      4) Optionally applies log2 transform to intensities.
      5) Removes rows with missing protein identifiers and rows with all-NA intensities.
      6) Collapses duplicated proteins by mean across samples (NA-safe).
      7) Returns an AnnData with samples in `.obs` and proteins in `.var`,
         plus convenience layers/metrics (median normalization; optional thresholded layers).

    Parameters
    ----------
    proteome_df : pandas.DataFrame
        Input table with **one row per protein** and columns containing protein annotations
        and sample intensities. Protein identifier column must be present
        (see `match_protein_column`).
    metadata_df : pandas.DataFrame
        Sample metadata with **one row per sample**. Must include the column named in
        `match_sample_column` that matches the sample columns in `proteome_df`.
    match_sample_column : str
        Column in `metadata_df` that contains sample IDs matching sample-intensity
        columns of `proteome_df`.
    match_protein_column : str
        Column in `proteome_df` containing the protein identifier to use as `.var_names`
        (e.g., gene symbol or accession).
    protein_annotation_columns : list of str
        Columns in `proteome_df` to carry over as protein annotations into `.var`.
        If `match_protein_column` is not included here, it will be appended for merging.
    need_log2 : bool, default True
        If True, apply log2 to intensity columns **after** numeric coercion.
        (Values are assumed positive if log2 is requested.)
    log2_intensity_threshold : float, optional
        If set, generates additional threshold-based layers (see Returns/Notes).

    Returns
    -------
    anndata.AnnData
        AnnData with:
          - `.X` : ndarray, shape (n_samples, n_proteins)
            Log2 intensities if `need_log2=True`, else raw (numeric-coerced) values.
          - `.obs` : DataFrame, shape (n_samples, *)
            Merged sample metadata (indexed by `match_sample_column`).
          - `.var` : DataFrame, shape (n_proteins, *)
            Protein annotations merged from `protein_annotation_columns`.
          - `.obsm["number_detection_sample"]` : ndarray, shape (n_samples,)
            Count of non-missing proteins per sample.
          - `.obsm["missing_rate_sample"]` : ndarray, shape (n_samples,)
            Fraction of missing values per sample.
          - `.varm["missing_rate_gene"]` : ndarray, shape (n_proteins,)
            Fraction of missing values per protein.
          - `.varm["median_intensity"]` : ndarray, shape (n_proteins,)
            Median intensity per protein (NaN-robust).
          - `.varm["sum_intensity"]` : ndarray, shape (n_proteins,)
            log2( sum( 2**X ) ) across samples (i.e., sum in linear space, then log2).
          - `.layers["median_norm"]` : ndarray, shape (n_samples, n_proteins)
            Sample-wise median-centered version of `.X`.
          - If `log2_intensity_threshold` is provided:
              * `.layers["threshold_controlled"]` :
                log2( 2**X + 2**threshold ) — i.e., linear add then log2, NaNs filled to the threshold value.
          - `.uns["intensity_threshold"]` : float or None
            The threshold used, or None.

    Notes
    -----
    - Unknown columns in `proteome_df` (not in `protein_annotation_columns`, not in
      sample names) are dropped with a warning.
    - Commas in numeric fields are replaced by periods before conversion.
    - Duplicate protein IDs are averaged (mean across samples, ignoring NaNs).
    - `.X` is (samples × proteins) because the grouped table is transposed.
    - One-dimensional per-sample/per-protein metrics are stored in `.obsm` / `.varm`
      as 1D arrays for convenience.
    - Threshold-based layers are created only if `log2_intensity_threshold` is set.

    Raises
    ------
    KeyError
        If `match_protein_column` or any of `protein_annotation_columns` are missing.
    Warning
        Logs warnings when samples in `metadata_df` are not present in `proteome_df`,
        when unknown columns are dropped, or when protein IDs are missing.

    Examples
    --------
    >>> adata = proteome_2_anndata(
    ...     proteome_df=my_proteome_df,
    ...     metadata_df=my_metadata_df,
    ...     match_sample_column="Sample",
    ...     match_protein_column="Gene",
    ...     protein_annotation_columns=["Gene", "PG.Genes", "PG.ProteinGroups"]
    ... )

    Archived layers (deleted in newer versions)
    ---------
    * `.layers["threshold_removed"]` :
      `.X` with entries < threshold set to NaN.
    * `.layers["threshold_added"]` :
      log2( 2**X + 2**threshold ) — i.e., linear add then log2.
    * `.layers["median_norm_threshold_added"]` :
      Sample-median-centered version of `threshold_added`.
    * `.layers["median_norm_threshold_controlled"]` :
      Sample-median-centered version of `threshold_controlled`.
    """

    # 1. Identify samples
    sample_names = [sample for sample in metadata_df[match_sample_column] if sample in proteome_df.columns]
    n_sample_excluded = metadata_df.shape[0] - len(sample_names)
    if n_sample_excluded > 0:
        logger.warning(f"{n_sample_excluded} samples are in the metadata but not in proteome_df. Double-check this!")

    # 2. Filter proteome_df columns
    df_to_adata = proteome_df.copy()
    mismatch_columns = proteome_df.shape[1] - (len(sample_names) + len(protein_annotation_columns) + 1)  # +1 for match_protein_column
    if mismatch_columns > 0:
        logger.warning(f"{mismatch_columns} columns in proteome_df are not found in metadata or protein_annotation_columns and will be removed.")
    df_to_adata = df_to_adata.filter(protein_annotation_columns + [match_protein_column] + sample_names)

    # 3. Remove rows with missing protein names
    n_empty_protein = df_to_adata[match_protein_column].isna().sum()
    if n_empty_protein > 0:
        logger.warning(f"{n_empty_protein} rows have no protein names (NaN) and will be removed.")
        df_to_adata = df_to_adata[df_to_adata[match_protein_column].notna()]

    # 4. Save protein-level info
    if match_protein_column in protein_annotation_columns:
        protein_info = df_to_adata[protein_annotation_columns].drop_duplicates(match_protein_column)
    else:
        protein_info = df_to_adata[protein_annotation_columns + [match_protein_column]].drop_duplicates(match_protein_column)

    # 5. Clean sample columns: replace commas, convert to numeric
    df_to_adata[sample_names] = df_to_adata[sample_names].replace(',', '.', regex=True)
    df_to_adata[sample_names] = df_to_adata[sample_names].apply(pd.to_numeric, errors='coerce')

    # 6. Remove rows where all sample columns are NA
    original_rows = df_to_adata.shape[0]
    df_to_adata = df_to_adata.dropna(subset=sample_names, how='all')
    removed_rows = original_rows - df_to_adata.shape[0]
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows where all sample columns were NA.")

    # 7. Log2 transformation if needed
    if need_log2:
        logger.info("Applying log2 transformation to intensity data.")
        df_to_adata[sample_names] = np.log2(df_to_adata[sample_names])

    # 8. Handle duplicate proteins by averaging
    n_duplicated_protein = df_to_adata.shape[0] - df_to_adata[match_protein_column].nunique()
    if n_duplicated_protein > 0:
        logger.info(f"{n_duplicated_protein} rows have duplicate protein names. Averaging intensities for duplicates (ignoring NA).")

    # Group by protein name, average across samples
    df_grouped = df_to_adata.groupby(match_protein_column)[sample_names].mean()

    # 9. Create AnnData object
    adata = ad.AnnData(df_grouped.transpose())

    # 10. Add sample metadata
    adata.obs[match_sample_column] = adata.obs_names
    adata.obs = adata.obs.reset_index().merge(metadata_df, how='left', on=match_sample_column).set_index(match_sample_column)

    # 11. Add protein metadata
    adata.var = adata.var.merge(protein_info, how='left', on=match_protein_column).set_index(match_protein_column)

    # 12. Calculate missingness
    adata.obsm["number_detection_sample"] = adata.X.shape[1] - np.isnan(adata.X).sum(axis=1)
    adata.obsm["missing_rate_sample"] = np.isnan(adata.X).sum(axis=1) / adata.X.shape[1]
    adata.varm["missing_rate_gene"] = np.isnan(adata.X).sum(axis=0) / adata.X.shape[0]

    # 13. Calculate median and sum intensity
    adata.varm["median_intensity"] = np.nanmedian(adata.X, axis=0)
    adata.varm["sum_intensity"] = np.log2(np.nansum(2 ** adata.X, axis=0))

    # 14. Threshold-controlled layers, here missing values are replaced with the threshold value
    if log2_intensity_threshold is not None:
        logger.info(f"Applying threshold control with log2 intensity threshold = {log2_intensity_threshold}")

        threshold_added = np.log2(2 ** adata.X + 2 ** log2_intensity_threshold)
        threshold_controlled = np.nan_to_num(threshold_added, nan=log2_intensity_threshold)
        adata.layers["threshold_controlled"] = threshold_controlled

        adata.uns["intensity_threshold"] = log2_intensity_threshold
    else:
        logger.info("No log2 intensity threshold specified. Threshold layers not computed.")
        adata.uns["intensity_threshold"] = None

    logger.info("AnnData object successfully created and returned.")
    return adata

def get_metadata_from_colname(df_proteome: pd.DataFrame, metadata_colnames: list):
    """
    Parse sample-level metadata from Spectronaut-style column names.

    This helper targets Spectronaut outputs where raw intensity columns end with
    ``.raw.PG.Quantity`` and may be prefixed by a numeric tag like ``[12] ``.
    It strips those decorations, asserts uniqueness, then splits the cleaned
    sample name on underscores to derive metadata fields.

    Parameters
    ----------
    df_proteome : pandas.DataFrame
        Spectronaut output table containing columns with suffix ``.raw.PG.Quantity``.
    metadata_colnames : list of str
        Names to assign to the underscore-delimited tokens parsed from each sample
        name (e.g., ``["Project", "Condition", "Batch"]``). The number of provided
        names should match the number of tokens in the sample naming scheme.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per sample and columns named by `metadata_colnames`,
        plus a column ``"sample"`` containing the cleaned sample identifier.
        The index is the sample identifier as parsed from the column names.

    Notes
    -----
    - Cleaning steps:
        * Remove leading ``[\\d+] `` tags.
        * Drop the ``.raw.PG.Quantity`` suffix.
    - The function assumes **consistent underscore-delimited structure** across all
      samples. If token counts differ, pandas will create extra unnamed columns
      (0, 1, 2, …); only the first ``len(metadata_colnames)`` will be renamed.
    - Ensure the count of `metadata_colnames` matches your naming scheme.

    Raises
    ------
    ValueError
        If cleaned sample names are not unique.

    Examples
    --------
    >>> meta = get_metadata_from_colname(df_proteome, ["Project", "Condition", "Replicate"])
    >>> meta.head()
    """

    sample_name = df_proteome.filter(regex=".raw.PG.Quantity").columns.tolist()
    sample_name = [re.sub(r"\[\d+\] ", "", sample) for sample in sample_name]
    sample_name = [sample.replace(".raw.PG.Quantity", "") for sample in sample_name]
    if (len(set(sample_name)) != len(sample_name)):
        raise ValueError("Sample names are not unique. Please check the input file.")
    metadata = pd.DataFrame({sample: sample.split("_") for sample in sample_name}).T
    metadata.rename(columns={old_col: new_col for old_col, new_col in zip(metadata.columns, metadata_colnames)}, inplace=True)
    metadata["sample"] = metadata.index
    return metadata

def spectronaut_2_anndata(proteome_df: pd.DataFrame, 
                          metadata_df: pd.DataFrame, 
                          match_sample_column: str = "sample",
                          match_protein_column: str = "Gene name", 
                          protein_annotation_columns: list = ["Gene name", "PG.Genes", "PG.ProteinGroups", "PG.ProteinDescriptions"],
                          log2_intensity_threshold: float = 2,
                          proteome_2_anndata: callable = None):
    """
    Convert a Spectronaut export into an :class:`anndata.AnnData`, with optional threshold layers.

    Workflow:
      1) Clean column names (strip leading ``[\\d+] `` tags).
      2) Insert primary ``"Gene name"`` from the first token of ``"PG.Genes"``.
      3) Keep raw-intensity and protein-annotation columns.
      4) Rename sample columns by removing the ``.raw.PG.Quantity`` suffix.
      5) Delegate to `proteome_2_anndata` to build the AnnData (includes log2 transform by default).
      6) Create/overwrite a convenience layer ``"threshold_removed"`` where values below
         `log2_intensity_threshold` are set to NaN.
      7) Update per-protein missingness for the thresholded layer.
      8) Store a copy of `.X` into `.layers["raw_log2"]`.

    Parameters
    ----------
    proteome_df : pandas.DataFrame
        Spectronaut output. Must contain raw intensity columns ending with
        ``.raw.PG.Quantity`` as well as protein annotations (e.g., ``PG.Genes``).
    metadata_df : pandas.DataFrame
        Sample metadata; must contain `match_sample_column`.
    match_sample_column : str, default "sample"
        Column in `metadata_df` matching the cleaned sample names.
    match_protein_column : str, default "Gene name"
        Protein identifier column to use as `.var_names`.
    protein_annotation_columns : list of str, optional
        Protein-annotation columns to retain/merge into `.var`.
    log2_intensity_threshold : float, default 2
        Threshold in **log2 space** used to generate ``"threshold_removed"``.
    proteome_2_anndata : callable
        Function used to build AnnData from the cleaned table. It must accept
        `(proteome_df, metadata_df, match_sample_column, match_protein_column, protein_annotation_columns, ...)`
        and return an AnnData. Typically this is the `proteome_2_anndata` defined above.

    Returns
    -------
    anndata.AnnData
        AnnData with:
          - `.X` : ndarray, shape (n_samples, n_proteins)
            Log2 intensities (assuming the delegated function’s default `need_log2=True`).
          - `.obs`, `.var` : sample/protein metadata.
          - `.layers["threshold_removed"]` :
            Copy of `.X` with values < `log2_intensity_threshold` set to NaN (overwrites if already present).
          - `.varm["missing_rate_gene_threshold_removed"]` :
            Missing fraction per protein after thresholding.
          - `.layers["raw_log2"]` :
            Copy of `.X` (for convenience/reference).

    Notes
    -----
    - If the delegated `proteome_2_anndata` also constructs threshold-based layers,
      this function **overwrites** ``"threshold_removed"`` to ensure it matches the
      threshold applied directly to `.X`.
    - Cleaning removes only the **prefix** ``[\\d+] `` and the **suffix**
      ``.raw.PG.Quantity``; the remaining sample token is used as the sample ID.

    Raises
    ------
    ValueError
        If `proteome_2_anndata` is not provided.

    Examples
    --------
    >>> adata = spectronaut_2_anndata(
    ...     proteome_df=spectronaut_df,
    ...     metadata_df=meta_df,
    ...     match_sample_column="sample",
    ...     match_protein_column="Gene name",
    ...     protein_annotation_columns=["Gene name","PG.Genes","PG.ProteinGroups","PG.ProteinDescriptions"],
    ...     log2_intensity_threshold=2,
    ...     proteome_2_anndata=proteome_2_anndata
    ... )
    """
    if proteome_2_anndata is None:
        raise ValueError("Please provide the 'proteome_2_anndata' function as an argument.")

    df = proteome_df.copy()

    # 1. Clean column names
    df.rename({col: re.sub(r"\[\d+\] ", "", col) for col in df.columns}, axis=1, inplace=True)
    logger.info("Cleaned column names by removing numeric prefixes.")

    # 2. Insert primary gene name
    df.insert(0, "Gene name", df["PG.Genes"].str.split(";").str[0])
    logger.info("Extracted primary gene name from 'PG.Genes' column.")

    # 3. Filter relevant columns
    df = df.filter(regex=".raw.PG.Quantity|Gene name|PG.ProteinGroups|PG.Genes|PG.ProteinDescriptions", axis=1)
    logger.info("Filtered relevant columns (raw intensities and protein annotations).")

    # 4. Rename sample columns to remove suffixes
    df.rename(columns={sample: sample.replace(".raw.PG.Quantity", "") for sample in df.columns}, inplace=True)
    logger.info("Removed '.raw.PG.Quantity' suffixes from sample columns.")

    # 5. Convert to AnnData
    adata_df = proteome_2_anndata(
        proteome_df=df,
        metadata_df=metadata_df,
        match_sample_column=match_sample_column,
        match_protein_column=match_protein_column,
        protein_annotation_columns=protein_annotation_columns,
        log2_intensity_threshold=log2_intensity_threshold
    )
    logger.info("Converted proteomic data to AnnData.")

    # 6. Create 'threshold_removed' layer
    threshold_removed = adata_df.X.copy()
    threshold_removed[threshold_removed < log2_intensity_threshold] = np.nan
    adata_df.layers["threshold_removed"] = threshold_removed
    logger.info(f"Created 'threshold_removed' layer with values below {log2_intensity_threshold} set to NaN.")

    # 9. Update missing rate
    adata_df.varm["missing_rate_gene_threshold_removed"] = np.isnan(adata_df.layers["threshold_removed"]).sum(axis=0) / adata_df.X.shape[0]

    # 10. Store raw log2-transformed data in a layer
    adata_df.layers["raw_log2"] = adata_df.X.copy()
    logger.info("Stored raw log2-transformed intensities in 'raw_log2' layer.")

    logger.info("Spectronaut data successfully converted to AnnData with threshold layers.")
    return adata_df

