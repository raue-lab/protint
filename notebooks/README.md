# ProtInt Proteomics Workflow — Notebook Documentation

## Workflow purpose

This notebook series prepares tumor and cancer-cell-line proteomics data, merges the datasets, generates a Celligner integration, tunes and tests ProtInt, benchmarks integration methods, and analyzes protein/pathway changes after projecting cell lines onto the tumor domain.

## Recommended execution order

1. `01.1_process_procan_depmapsanger.ipynb`
2. `01.2_process_FDL_proteome.ipynb`
3. `01.3_merge_data.ipynb`
4. `03.0_celligner_integration.ipynb`
5. `03.1_protint_tune_dropout_parameter.ipynb`
6. `03.2_protint_tune_adv_cyc_weight.ipynb`
7. `03.3_protint_test_package.ipynb`
8. `03.4_benchmarking.ipynb`
9. `03.5_cell_line_projection_analysis.ipynb`

`03.0` must run before `03.4` because the benchmarking notebook imports its Celligner-transformed matrix. The two tuning notebooks (`03.1` and `03.2`) are exploratory and can be skipped after final hyperparameters have been selected.

## Shared configuration

The notebooks load `config.local.yaml`. Across the workflow, the required keys are:

- `code_dir`
- `output_data_dir`
- `output_plot_dir`
- `cell_line_data_dir`, `cell_line_data_filename`, `cell_line_metadata_filename`
- `tumor_data_dir`, `tumor_data_filename`

The notebooks use `../../config/config.local.yaml`. Run them from the expected repository locations or replace these paths with absolute paths.

---

## `01.1_process_procan_depmapsanger.ipynb`

### Purpose

Convert the ProCan-DepMapSanger (https://pmc.ncbi.nlm.nih.gov/articles/PMC9387775/#da0010) cancer-cell-line proteomics matrix into an AnnData object suitable for merging with the FDL tumor data.

### Main inputs

Configured tab-separated files:

- ProCan proteomics matrix.
- ProCan sample metadata file.

### Core processing

- Reads intensity data with comma decimal notation.
- Parses project and cell-line labels from the proteomics sample headers.
- Transposes the matrix to samples × proteins.
- Parses UniProt ID, gene symbol, and organism from protein identifiers.
- Calls `proteome_2_anndata(..., need_log2=False)` for QC.
- Generates QC, sample-intensity, and completeness plots.

### Main output

```text
<output_data_dir>/processed_proteomics_data/PanCancer_Goncalves2022_cancer_cell_lines.h5ad
```

### Execution note

The notebook loads the external metadata table and then replaces `metadata_df` with labels parsed from the proteomics headers. The recreated table contains a column named `cell line`, while `proteome_2_anndata` is called with `match_sample_column="Cell_line"`. Confirm the expected column name in `proteome_2_anndata`; otherwise rename the column or change the argument before execution.

---

## `01.2_process_FDL_proteome.ipynb`

### Purpose

Convert the FDL cohort-1 tissue proteomics data (https://doi.org/10.1158/2159-8290.CD-24-1488https://doi.org/10.1158/2159-8290.CD-24-1488) into an AnnData object for joint analysis with ProCan cell lines.

### Main input

A configured tab-separated FDL proteomics matrix containing sample-level annotation columns.

### Core processing

- Extracts `SampleID`, cancer type, tissue type, and cancer subtype into `metadata_df`.
- Removes these metadata columns from the intensity matrix.
- Transposes the matrix to samples × proteins and uses UniProt IDs as feature identifiers.
- Calls `proteome_2_anndata(..., need_log2=False)` for QC.
- Recreates the object using `log2_intensity_threshold=-2.5`.
- Generates QC and sample-completeness plots.
- Retains tumor, adjacent-normal, and healthy samples; tumor-only filtering occurs during merging.

### Main output

```text
<output_data_dir>/processed_proteomics_data/PanCancer_Cai2025_FDL.h5ad
```

---

## `01.3_merge_data.ipynb`

### Purpose

Create the joint FDL tumor and ProCan cell-line AnnData objects used for ProtInt training and downstream analyses.

### Main inputs

- `PanCancer_Cai2025_FDL.h5ad`
- `PanCancer_Goncalves2022_cancer_cell_lines.h5ad`
- Manual tissue/cancer annotation CSV under `00_manual_mapping/FDL_ProCan/`

### Core processing

- Removes non-tumor FDL samples and proteins missing in all FDL samples.
- Harmonizes observation fields and joint tissue annotations.
- Restricts both sources to shared UniProt features.
- Stores observed/missing status in `layers["detected"]`, then replaces missing `.X` values with zero.
- Concatenates FDL and ProCan samples.
- Creates a second object containing tissue types with at least 20 samples in both sources.
- Removes proteins not detected in both sources in the tissue-20 object.
- Fits source-specific probabilistic dropout curves.
- Generates intensity, missingness, tissue-count, dropout-fit, and UMAP diagnostics.

### Main outputs

- `1_1_9_ProCan_FDL_PanCancer_merged.h5ad`
- `1_1_9_ProCan_FDL_PanCancer_merged_tissue20.h5ad`

Both are written under:

```text
<output_data_dir>/processed_proteomics_data/
```


---

## `03.0_celligner_integration.ipynb`

### Purpose

Adapt Celligner to integrate ProCan cancer-cell-line and FDL tumor proteomics, producing the Celligner-corrected matrix used in the integration benchmark.

### Main input

```text
<output_data_dir>/processed_proteomics_data/1_1_9_ProCan_FDL_PanCancer_merged_tissue20.h5ad
```

The input is the merged, shared-feature dataset with missing values replaced by zero and only tissue groups represented by at least 20 samples in each source.

### Core processing

- Splits the merged AnnData object into `ProCan_cell_line` and `FDL_patient`.
- Fits `Celligner` on the ProCan cell-line expression matrix.
- Transforms the FDL patient matrix into the fitted Celligner space.
- Calls `computeMetricsForOutput()` and inspects generated DataFrame attributes.
- Annotates Celligner UMAP coordinates with data source and tissue type for inspection.
- Exports `my_celligner.combined_output` as the aligned sample-by-protein matrix.

### Main output

```text
<output_data_dir>/celligner_integration/transformed_data_celligner_FDL_Procan.csv
```

`03.4_benchmarking.ipynb` reads this exact file and aligns it to the merged AnnData sample and feature order.

### Environment requirement

Run in the dedicated Celligner environment with Python 3.9 and the `celligner` package installed. The notebook notes that the main analysis environment uses Python 3.11 and is not compatible with this Celligner setup.

### Execution notes

- Celligner was originally designed for transcriptomics; this notebook applies it to log2 proteomics with missing values replaced by zero.
- `combined_output` must retain all sample and protein identifiers expected by `03.4`; the benchmark uses strict `.loc[adata.obs_names, adata.var_names]` alignment.

---

## `03.1_protint_tune_dropout_parameter.ipynb`

### Purpose

Select the ProtInt dropout-loss weight and compare fixed versus trainable dropout-curve parameters (`rho`, `zeta`).

### Core processing

- Sweeps dropout weights from `0` to `100` across five seeds.
- Keeps adversarial and cycle-consistency weights at zero.
- Repeats the sweep with `--train_dropout`.
- Records final reconstruction loss.
- Trains one selected model and compares input, reconstruction, and projection.

### Main outputs

- Model/projection files under `protint_tune_dropout_parameter`.
- UMAPs, reconstruction diagnostics, and loss-versus-weight plots.

### Key settings

Hidden dimensions `128 64`; latent dimension `32`; batch size `512`; sweep epochs `3000`; KL weight `1e-6`.

---

## `03.2_protint_tune_adv_cyc_weight.ipynb`

### Purpose

Tune adversarial and cycle-consistency weights while monitoring reconstruction, source mixing, and tissue preservation.

### Core processing

- Fixes `dropout_weight=0.1`.
- Separately sweeps `src_adv_weight` and `cyc_weight` over five seeds.
- Projects cell lines onto `FDL_patient`.
- Records reconstruction loss, graph iLISI, and tissue cLISI at `k=15`.

### Main outputs

- `tune_adv_cyc_weight_results_fixed_dropout.csv`
- `tune_adv_cyc_weight_results_trainable_dropout.csv`
- UMAP, reconstruction, loss, iLISI, and cLISI plots.

### Execution note

In the intended trainable-dropout section, the adversarial sweep includes `--train_dropout`, but the cycle-consistency sweep does not. Add the flag if trainable dropout is required for both sweeps.

---

## `03.3_protint_test_package.ipynb`

### Purpose

Run an end-to-end ProtInt package test and evaluate whether projected cell lines preserve tissue-specific relationships to tumors.

### Core processing

- Trains ProtInt and plots training loss.
- Projects samples onto the `FDL_patient` domain.
- Reconstructs samples using their original source labels.
- Compares input, reconstruction, and projection.
- Classifies each patient from its five nearest projected cell lines in PCA space.
- Reports accuracy and confusion matrices.

### Main outputs

- Model and `projection.h5ad` under `protint_output`.
- Training, UMAP, reconstruction, and classification diagnostics.

---

## `03.4_benchmarking.ipynb`

### Purpose

Compare ProtInt against alternative integration and batch correction methods using UMAPs and LISI metrics.

### Methods

Original data, ProtInt, MOBER, Celligner, ComBat, limma, and Harmony.

### Core processing

- Runs stochastic methods across seeds `11–15`.
- Imports and strictly aligns the Celligner matrix generated by `03.0`.
- Applies ComBat, limma, and Harmony through project utilities.
- Computes graph iLISI and tissue cLISI at `k=15`.
- Reports mean and standard deviation for stochastic methods.

### Main outputs

- Corrected/projection `.h5ad` files.
- `result_iLISI.pkl`
- `result_cLISI_tissue.pkl`
- Method UMAPs, reconstruction distributions, and LISI comparisons.

### Dependencies

Requires the Celligner CSV generated by `03.0`, the MOBER CLI, and `harmonypy==0.0.10` (installed inside the notebook).

---

## `03.5_cell_line_projection_analysis.ipynb`

### Purpose

Identify protein and Reactome pathway changes introduced when ProtInt decodes cell lines as tumors.

### Core processing

- Reconstructs samples with original source labels.
- Aligns input, reconstruction, and projection objects.
- Maps UniProt accessions to gene symbols through BioMart.
- Compares paired cell-line projection versus reconstruction with limma.
- Requires proteins to be detected in both paired states for at least 30% of pairs.
- Runs Reactome GSEA with `fgseaMultilevel`, gene-set sizes `10–500`, and tissue-specific BH correction.
- Repeats analyses by tissue and across all cell lines.

Positive log fold change/t-statistic means higher abundance in projection; negative values mean higher abundance in reconstruction.

### Main outputs

- Per-tissue and all-cell-line limma/GSEA CSV files.
- Per-gene comparison plots.
- Volcano, GSEA bubble, pathway matrix, and NES heatmap figures.

### Additional dependencies

Python: `rpy2`, `gseapy`, `adjustText`.

R: `limma`, `fgsea`, `msigdbr`.

BioMart mapping requires network access unless a local mapping is supplied.

---

## Reproducibility notes

- The merged AnnData object must contain `obs["data_source"]`, `obs["joint tissue type"]`, and `layers["detected"]`.
- Feature order must remain consistent across training, reconstruction, and projection objects.
- `protint` and `mober` must be available in the active environment.
- Several subprocess calls use `check=False`; inspect return codes and generated files.
- Parameter and seed loops can overwrite model/projection outputs; preserve selected runs separately.
- The notebooks were reviewed for documentation but were not executed against the full data and software environment.
