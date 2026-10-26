#!/usr/bin/env nextflow
// Esboço DSL2 equivalente ao pipeline Snakemake (pipelines/snakemake).
// Reaproveita os mesmos scripts Python via linha de comando.
// Uso: nextflow run main.nf --amostras 'dados/*.h5ad' --resolucoes 0.5,1.0 -profile docker
nextflow.enable.dsl = 2

params.amostras   = "dados/*.h5ad"
params.resolucoes = "0.5,1.0"
params.min_genes  = 200
params.max_pct_mt = 15
params.n_hvg      = 2000
params.seed       = 0
params.outdir     = "resultados"

process QC {
    tag "$amostra"
    publishDir "${params.outdir}/${amostra}", mode: 'copy'
    input:  tuple val(amostra), path(h5ad)
    output: tuple val(amostra), path("qc.h5ad")
    script:
    """
    python - <<'PY'
import scanpy as sc
a = sc.read_h5ad("${h5ad}"); a.var_names_make_unique()
a.var["mt"] = a.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
a = a[(a.obs.n_genes_by_counts >= ${params.min_genes}) & (a.obs.pct_counts_mt < ${params.max_pct_mt})].copy()
sc.pp.filter_genes(a, min_cells=3); a.layers["counts"] = a.X.copy(); a.write_h5ad("qc.h5ad")
PY
    """
}

process NORMALIZAR {
    tag "$amostra"
    input:  tuple val(amostra), path(qc)
    output: tuple val(amostra), path("norm.h5ad")
    script:
    """
    python - <<'PY'
import scanpy as sc
a = sc.read_h5ad("${qc}")
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=${params.n_hvg}, flavor="seurat_v3", layer="counts")
sc.pp.pca(a, n_comps=30, mask_var="highly_variable", random_state=${params.seed})
sc.pp.neighbors(a, random_state=${params.seed}); a.write_h5ad("norm.h5ad")
PY
    """
}

process CLUSTER {
    tag "${amostra} res=${res}"
    publishDir "${params.outdir}/${amostra}", mode: 'copy'
    input:  tuple val(amostra), path(norm), val(res)
    output: tuple val(amostra), path("cluster_res${res}.csv")
    script:
    """
    python - <<'PY'
import scanpy as sc
a = sc.read_h5ad("${norm}")
sc.tl.leiden(a, resolution=${res}, random_state=${params.seed}, flavor="igraph", n_iterations=2, directed=False)
a.obs[["leiden"]].to_csv("cluster_res${res}.csv")
PY
    """
}

workflow {
    amostras   = Channel.fromPath(params.amostras).map { f -> tuple(f.baseName, f) }
    resolucoes = Channel.of(params.resolucoes.tokenize(',')).flatten()
    normalizado = QC(amostras) | NORMALIZAR
    CLUSTER(normalizado.combine(resolucoes))
}
