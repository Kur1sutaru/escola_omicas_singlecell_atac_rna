import scanpy as sc
a = sc.read_h5ad(snakemake.input[0])
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=snakemake.params.n_hvg, flavor="seurat_v3", layer="counts")
sc.pp.pca(a, n_comps=30, mask_var="highly_variable", random_state=snakemake.params.seed)
sc.pp.neighbors(a, random_state=snakemake.params.seed)
a.write_h5ad(snakemake.output[0])
