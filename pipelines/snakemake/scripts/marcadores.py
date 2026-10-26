import scanpy as sc, pandas as pd
a = sc.read_h5ad(snakemake.input.norm)
a.obs["leiden"] = pd.read_csv(snakemake.input.cl, index_col=0)["leiden"].astype(str).astype("category").values
sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon")
sc.get.rank_genes_groups_df(a, None).groupby("group").head(5).to_csv(snakemake.output[0], index=False)
