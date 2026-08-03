# Here are functions for plotting proteome data in anndata format.
## These plots are intended for initial exploration of the data, such as checking sample completeness, sample intensities, and quality control metrics.

def plot_qc(adata,
            layer: str=None,
            xlab: str="Proteins, ordered by sum of intensity",
            ylab: str="Log2 intensity"):
    """
    Plot a QC scatterplot of protein intensities across genes in an AnnData object.

    Each point represents a single protein-sample measurement, with x-axis proteins ordered by
    decreasing total intensity and colored by their missing rate.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (typically from single-cell or spatial proteomics).
        The `.var` must include `sum_intensity_gene`, and `.varm` must include `missing_rate_gene`.
    
    layer : str, optional
        The layer of the AnnData object to use for intensity values.
        If None, the main `.X` matrix is used.

    xlab : str, default="Proteins, ordered by sum of intensity"
        Label for the x-axis.

    ylab : str, default="Log2 intensity"
        Label for the y-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.

    Notes
    -----
    - Proteins (genes) are sorted by `sum_intensity_gene`, assumed to be in `adata.varm`.
    - The x-axis ticks are removed for clarity.
    - If the specified `layer` is not found, the function prints an error and returns `None`.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(1,1, figsize=(8,7))
    if layer is None:
        df_plot = adata.to_df().stack().reset_index()
        df_plot.columns = ["sample", "Gene_Symbol", "intensity"]
    else:
        try:
            df_plot = adata.to_df(layer=layer).stack().reset_index()
            df_plot.columns = ["sample", "Gene_Symbol", "intensity"]
        except KeyError:
            print("Specified layer not found in the anndata object. Please double-check this!")
            return
    df_plot = df_plot.merge(pd.DataFrame({"Gene_Symbol": adata.var_names,
                                        "missing_rate_gene": adata.varm["missing_rate_gene"]}),
                            on="Gene_Symbol", how="left")
    gene_order = pd.DataFrame({"Gene_Symbol": adata.var_names,
                            "sum_intensity": adata.varm["sum_intensity"]})
    gene_order = gene_order.sort_values('sum_intensity', ascending=False)['Gene_Symbol']
    df_plot['Gene_Symbol'] = pd.Categorical(df_plot['Gene_Symbol'], categories=gene_order, ordered=True)
    sns.scatterplot(data=df_plot, x="Gene_Symbol", y="intensity", hue="missing_rate_gene", ax=ax)

    plt.legend(bbox_to_anchor=(1.1,0.5), title="missing rate across samples")
    ax.set_xticks([])

    plt.ylabel(ylab)
    plt.xlabel(xlab)
    return ax


def boxplot_sample_intensities(adata, layer=None, color_by=None, figsize=(16, 6)):
    """
    Plot boxplots of intensities for each sample (column) in the specified layer of an AnnData object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from scipy import sparse

    # Extract data matrix
    X = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(X):
        X = X.A  # to dense

    # var x obs -> genes as rows, samples as columns
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names).T

    # Long format
    melted = df.melt(var_name="Sample", value_name="Intensity")

    # Optional hue
    palette = None
    if color_by is not None:
        sample_labels = adata.obs[color_by].astype("string")
        # align to df columns (samples)
        sample_labels = sample_labels.reindex(df.columns)
        melted["Label"] = melted["Sample"].map(sample_labels)
        # build a palette only for present (non-missing) labels
        levels = pd.unique(melted["Label"].dropna())
        palette = sns.color_palette("Set2", len(levels))
    else:
        melted["Label"] = None

    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=melted,
        x="Sample", y="Intensity",
        hue=("Label" if color_by else None),
        palette=palette,
        dodge=False,
        showfliers=True,                       # <-- show outliers
        flierprops=dict(marker="o", markersize=2, alpha=0.35)  # style outliers
    )

    # tidy axes
    ax.set_xticklabels([])  # hide many sample labels
    ax.set_xlabel("Samples")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Boxplot of intensities per sample (layer: {layer})")
    plt.tight_layout()
    return ax


def plot_sample_completeness(adata, layer=None, color_by=None, figsize=(14, 5)):
    """
    Plot the completeness (fraction of non-missing values) for each sample (row) in the specified layer of an AnnData object.
    
    Parameters:
        adata: AnnData object
        layer: str or None, name of the layer to use for intensities (default: adata.X)
        color_by: str or None, column in adata.obs to color the bars by (default: None)
        figsize: tuple, size of the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Extract data
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object.")
        data = adata.layers[layer]
    else:
        data = adata.X

    # Compute completeness per sample (row)
    completeness = np.sum(~np.isnan(data), axis=1) / data.shape[1]
    df = pd.DataFrame({
        'Sample': adata.obs_names,
        'Completeness': completeness
    })

    # Add color_by if specified
    if color_by is not None:
        df['Label'] = adata.obs.loc[df['Sample'], color_by].values
        palette = sns.color_palette("Set2", df['Label'].nunique())
    else:
        df['Label'] = None
        palette = None

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x='Sample', y='Completeness', hue='Label' if color_by else None,
        data=df, dodge=False, palette=palette
    )
    ax.set_xticklabels([])
    ax.set_xlabel('Samples')
    ax.set_ylabel('Completeness (fraction non-missing)')
    ax.set_ylim(0, 1)
    plt.title(f'Sample completeness (layer: {layer})')
    plt.tight_layout()
    return ax