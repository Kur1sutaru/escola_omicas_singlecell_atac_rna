import shutil, scanpy as sc
if snakemake.params.entrada:
    shutil.copy(snakemake.params.entrada, snakemake.output[0])
else:
    sc.datasets.pbmc3k().write_h5ad(snakemake.output[0])
