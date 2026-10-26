import scanpy as sc
a = sc.read_h5ad(snakemake.input[0])
a.var_names_make_unique()
a.var["mt"] = a.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
n0 = a.n_obs
a = a[(a.obs.n_genes_by_counts >= snakemake.params.min_genes) & (a.obs.pct_counts_mt < snakemake.params.max_pct_mt)].copy()
sc.pp.filter_genes(a, min_cells=3)
a.layers["counts"] = a.X.copy()
a.write_h5ad(snakemake.output[0])
open(snakemake.log[0], "w").write(f"{n0} -> {a.n_obs} células\n")
