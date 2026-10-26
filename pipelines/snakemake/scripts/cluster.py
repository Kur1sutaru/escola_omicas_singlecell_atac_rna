import scanpy as sc
a = sc.read_h5ad(snakemake.input[0])
res = float(snakemake.wildcards.res)
sc.tl.leiden(a, resolution=res, random_state=snakemake.params.seed, flavor="igraph", n_iterations=2, directed=False)
a.obs[["leiden"]].to_csv(snakemake.output[0])
