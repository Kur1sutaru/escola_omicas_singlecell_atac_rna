# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   accelerator: GPU
#   colab:
#     provenance: []
#     gpuType: T4
# ---

# %% [markdown]
# # Aula 2 · Parte A: Multiome (RNA + ATAC pareados)
# ## MuData → WNN → MultiVI → ligações pico–gene
#
# **Curso:** Análise Avançada de Dados de Single-cell RNA-seq, ATAC-seq e Multiome (10x Genomics) no Google Colab
# **Ministrante:** Dra. Cristal Villalba · [github.com/kur1sutaru](https://github.com/kur1sutaru)
#
# ---
#
# No Multiome 10x, **a mesma célula** (o mesmo núcleo) tem RNA e ATAC medidos. Isso muda o jogo:
# não precisamos "adivinhar" correspondências entre modalidades, mas precisamos **combinar** dois sinais
# com ruídos e escalas muito diferentes.
#
# | Etapa | Ferramenta | Pergunta |
# |---|---|---|
# | 1. Estrutura | **MuData / muon** | como guardar duas matrizes que compartilham células? |
# | 2. Processar cada modalidade | Scanpy (RNA), TF-IDF + LSI (ATAC) | |
# | 3. Grafo conjunto | **WNN** (`mu.pp.neighbors`) | quanto cada modalidade "pesa" em cada célula? |
# | 4. Espaço latente conjunto | **MultiVI** (scvi-tools) | modelo generativo das duas modalidades |
# | 5. Comparação | métricas de vizinhança | RNA só × ATAC só × WNN × MultiVI |
# | 6. Ligações pico–gene | correlação em metacélulas | qual enhancer regula qual gene? |
# | 7. Ponte com R | Signac/Seurat | equivalências de funções |
#
# **Dados:** *PBMC from a healthy donor, granulocytes removed through cell sorting (3k)*, 10x Genomics Multiome (ARC v2.0).
# **Runtime:** GPU T4 recomendado (MultiVI).

# %% [markdown]
# ## 0. Instalação 🔧

# %% tags=["colab-only"]
# %pip install -q scanpy==1.12.4 anndata==0.13.4 mudata==0.4.1 muon==0.1.9 scvi-tools==1.5.1 \
#     celltypist==1.7.1 scikit-learn==1.7.2 leidenalg==0.12.0 session-info2

# %% [markdown]
# ## 1. Imports e parâmetros ⚙️

# %%
import os, warnings, random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import mudata as md
import muon as mu
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(4, 4))

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
DATA = Path(os.environ.get("CURSO_DATA", "/content/data" if IN_COLAB else "data"))
OUT = Path(os.environ.get("CURSO_OUT", "/content/resultados" if IN_COLAB else "resultados"))
DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)

# %% tags=["parameters"]
SEED = 0
N_HVG = 3000
MIN_CELLS_PEAK = 0.02          # pico precisa estar aberto em ≥ 2% das células (para o MultiVI)
MAX_EPOCHS_MULTIVI = 300       # ~5 min em T4 (com early stopping)
JANELA = 100_000               # janela (pb) ao redor do gene para ligações pico–gene
N_METACELULAS = 150
RODAR_TOTALVI = False          # seção opcional (CITE-seq), baixa outro dataset
random.seed(SEED); np.random.seed(SEED)

import scvi, torch
scvi.settings.seed = SEED
print("scvi-tools", scvi.__version__, "| GPU:", torch.cuda.is_available())

# %% [markdown]
# ## 2. Dados 📥
# O `filtered_feature_bc_matrix.h5` do **cellranger-arc** traz as duas matrizes no mesmo arquivo
# (`feature_type` = *Gene Expression* ou *Peaks*). O `mu.read_10x_h5` separa as modalidades em um **MuData**.

# %% tags=["load"]
import urllib.request

def baixar(url, destino):
    destino = Path(destino)
    if not destino.exists():
        print("baixando", destino.name); urllib.request.urlretrieve(url, destino)
    return destino

BASE = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/"
h5 = baixar(BASE + "pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5",
            DATA / "pbmc3k_multiome_filtered.h5")
mdata = mu.read_10x_h5(h5, extended=False)

# %%
mdata.var_names_make_unique()
mdata

# %% [markdown]
# ### Anatomia de um MuData
# - `mdata.mod["rna"]` e `mdata.mod["atac"]` são **AnnData comuns**;
# - `mdata.obs` / `mdata.obsm` guardam o que é **compartilhado** (ex.: o UMAP do WNN);
# - colunas de cada modalidade aparecem em `mdata.obs` com prefixo (`rna:...`, `atac:...`) após `mdata.update()`.

# %%
rna, atac = mdata.mod["rna"], mdata.mod["atac"]
print("RNA :", rna.shape, "| ATAC:", atac.shape)
atac.var.head(3)

# %% [markdown]
# ## 3. QC por modalidade e interseção de células

# %%
rna.var["mt"] = rna.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].hist(np.log10(rna.obs["total_counts"] + 1), 60); ax[0].set_title("RNA log10(UMIs)")
ax[1].hist(rna.obs["pct_counts_mt"], 60); ax[1].set_title("RNA % mito")
ax[2].hist(np.log10(atac.obs["total_counts"] + 1), 60); ax[2].set_title("ATAC log10(contagens em picos)")
plt.tight_layout(); plt.show()

# %%
def limites_mad(x, nmads=4):
    x = np.log1p(x); m = np.median(x); d = np.median(np.abs(x - m))
    return np.expm1(m - nmads * d), np.expm1(m + nmads * d)

lo_r, hi_r = limites_mad(rna.obs["total_counts"])
lo_a, hi_a = limites_mad(atac.obs["total_counts"])
mu.pp.filter_obs(rna, "total_counts", lambda x: (x > lo_r) & (x < hi_r))
mu.pp.filter_obs(rna, "pct_counts_mt", lambda x: x < 20)
mu.pp.filter_obs(atac, "total_counts", lambda x: (x > lo_a) & (x < hi_a))
mu.pp.intersect_obs(mdata)                       # só células que passaram nas DUAS modalidades
rna, atac = mdata.mod["rna"], mdata.mod["atac"]
print(f"{mdata.n_obs} células após QC conjunto")

# %% [markdown]
# ## 4. Processar cada modalidade
# ### 4.1 RNA

# %%
rna.layers["counts"] = rna.X.copy()
sc.pp.filter_genes(rna, min_cells=5)
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, n_top_genes=N_HVG, flavor="seurat_v3", layer="counts")
sc.pp.pca(rna, n_comps=30, mask_var="highly_variable", random_state=SEED)
sc.pp.neighbors(rna, n_neighbors=15, random_state=SEED)
sc.tl.umap(rna, random_state=SEED)

# %% [markdown]
# ### 4.2 ATAC: TF-IDF + LSI (o mesmo que `RunTFIDF` + `RunSVD` do Signac)

# %%
atac.layers["counts"] = atac.X.copy()
sc.pp.filter_genes(atac, min_cells=10)
mu.atac.pp.tfidf(atac, scale_factor=1e4)
mu.atac.tl.lsi(atac, n_comps=31)
r1 = np.corrcoef(atac.obsm["X_lsi"][:, 0], np.log1p(atac.obs["total_counts"]))[0, 1]
print(f"corr(LSI_1, profundidade) = {r1:.2f} → descartado")
atac.obsm["X_lsi"] = atac.obsm["X_lsi"][:, 1:]
atac.varm["LSI"] = atac.varm["LSI"][:, 1:]
atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
sc.pp.neighbors(atac, use_rep="X_lsi", n_neighbors=15, random_state=SEED)
sc.tl.umap(atac, random_state=SEED)

# %% [markdown]
# ### 4.3 Rótulos de referência (CellTypist no RNA) 📥
# Usamos o RNA para dar rótulos às células; eles servirão para **avaliar** as integrações.

# %% tags=["load"]
import celltypist
from celltypist import models
models.download_models(model=["Immune_All_High.pkl"])
modelo_grosso = models.Model.load("Immune_All_High.pkl")

# %%
pred = celltypist.annotate(rna, model=modelo_grosso, majority_voting=False)
rna.obs["cell_type"] = pred.predicted_labels["predicted_labels"].astype("category")
mdata.obs["cell_type"] = rna.obs["cell_type"].reindex(mdata.obs_names).values
atac.obs["cell_type"] = mdata.obs["cell_type"].values
sc.pl.umap(rna, color="cell_type", title="RNA", show=False)
sc.pl.umap(atac, color="cell_type", title="ATAC (LSI)")

# %% [markdown]
# ## 5. WNN: *weighted nearest neighbors*
#
# Ideia (Hao et al., *Cell* 2021): para cada célula, quão bem a modalidade **A** prediz os vizinhos da modalidade **B**?
# A modalidade que "explica melhor" a vizinhança daquela célula ganha mais **peso**.
# - pesos **por célula**, não globais: T CD4 × CD8 se separam melhor pelo RNA; certos estados de cromatina, pelo ATAC;
# - o resultado é um **grafo** (não um *embedding*): serve para UMAP e clusterização.

# %%
mdata.update()
mu.pp.neighbors(mdata, key_added="wnn", add_weights_to_modalities=True, random_state=SEED)
mu.tl.umap(mdata, neighbors_key="wnn", random_state=SEED)
sc.tl.leiden(mdata, neighbors_key="wnn", key_added="leiden_wnn", resolution=1.0, random_state=SEED,
             flavor="igraph", n_iterations=2, directed=False)
mdata.obsm["X_wnn_umap"] = mdata.obsm["X_umap"]
mu.pl.embedding(mdata, basis="X_wnn_umap", color=["cell_type", "leiden_wnn"], wspace=0.5)

# %%
# com add_weights_to_modalities=True, o peso de cada célula fica em mdata.mod[m].obs["mod_weight"]
pesos = pd.DataFrame({"peso_RNA": rna.obs["mod_weight"].values,
                      "cell_type": mdata.obs["cell_type"].values}, index=mdata.obs_names)
ax = pesos.boxplot(column="peso_RNA", by="cell_type", rot=90, figsize=(7, 3.5), grid=False)
plt.suptitle(""); plt.title("Peso do RNA no WNN, por tipo celular"); plt.ylabel("peso RNA"); plt.show()

# %% [markdown]
# ## 6. MultiVI: modelo generativo das duas modalidades
#
# O **MultiVI** (Ashuach et al., *Nat Methods* 2023) é um VAE com dois *encoders* (RNA: binomial negativa;
# ATAC: Bernoulli) que se encontram num **espaço latente comum**. Vantagens:
# - integra células **pareadas e não pareadas** (só RNA ou só ATAC) no mesmo espaço;
# - corrige lote (`batch_key`);
# - **imputa** a modalidade ausente.
#
# Para caber no Colab, usamos HVGs e picos abertos em ≥ `MIN_CELLS_PEAK` das células.

# %%
picos_ok = np.asarray((atac.layers["counts"] > 0).mean(0)).ravel() >= MIN_CELLS_PEAK
mvi = md.MuData({
    "rna": rna[:, rna.var["highly_variable"]].copy(),
    "atac": atac[:, picos_ok].copy(),
})
print(mvi)

scvi.model.MULTIVI.setup_mudata(mvi, rna_layer="counts", atac_layer="counts",
                                modalities={"rna_layer": "rna", "atac_layer": "atac"})
modelo_mvi = scvi.model.MULTIVI(mvi, n_genes=mvi.mod["rna"].n_vars, n_regions=mvi.mod["atac"].n_vars)
modelo_mvi.train(max_epochs=MAX_EPOCHS_MULTIVI, early_stopping=True, accelerator="auto")

# %%
mdata.obsm["X_multivi"] = modelo_mvi.get_latent_representation(mvi)
sc.pp.neighbors(mdata, use_rep="X_multivi", key_added="multivi", random_state=SEED)
sc.tl.umap(mdata, neighbors_key="multivi", random_state=SEED)
mdata.obsm["X_multivi_umap"] = mdata.obsm["X_umap"]
mu.pl.embedding(mdata, basis="X_multivi_umap", color="cell_type", title="MultiVI")

# %% [markdown]
# ## 7. Qual representação separa melhor os tipos celulares?
# Usamos a **pureza da vizinhança** (fração de vizinhos com o mesmo rótulo) em cada grafo.
# Atenção: os rótulos vieram do **RNA**, então o RNA tem vantagem. Interprete o resultado relativo.

# %%
def pureza(conn, labels):
    G = conn.tocsr(); lab = pd.Categorical(labels).codes
    return np.mean([(lab[G.indices[G.indptr[i]:G.indptr[i + 1]]] == lab[i]).mean()
                    for i in range(G.shape[0]) if G.indptr[i + 1] > G.indptr[i]])

lab = mdata.obs["cell_type"].astype(str).values
grafos = {
    "RNA (PCA)": rna.obsp["connectivities"],
    "ATAC (LSI)": atac.obsp["connectivities"],
    "WNN": mdata.obsp["wnn_connectivities"],
    "MultiVI": mdata.obsp["multivi_connectivities"],
}
comp = pd.Series({k: pureza(v, lab) for k, v in grafos.items()}, name="pureza kNN").to_frame()
comp.style.format("{:.3f}").background_gradient(cmap="Greens")

# %%
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax_, (nome, emb, obj) in zip(axes, [("RNA", "X_umap", rna), ("ATAC", "X_umap", atac),
                                         ("WNN", "X_wnn_umap", mdata), ("MultiVI", "X_multivi_umap", mdata)]):
    sc.pl.embedding(obj, basis=emb, color="cell_type", ax=ax_, show=False, title=nome,
                    legend_loc=None if nome != "MultiVI" else "right margin", size=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### (Opcional) E o totalVI?
# O **totalVI** é o "irmão" do MultiVI para **CITE-seq** (RNA + proteínas de superfície, ADT), **não** para ATAC.
# A lógica é a mesma (VAE conjunto), mas a proteína tem fundo (*background*) que o modelo separa do sinal.
# Use quando o seu experimento for 10x *Feature Barcoding* com anticorpos (ou *TEA-seq/DOGMA-seq*, que tem as três modalidades).

# %% tags=["optional"]
if RODAR_TOTALVI:
    cite = scvi.data.pbmcs_10x_cite_seq(save_path=str(DATA))          # 2 lotes, RNA + 14 proteínas
    sc.pp.highly_variable_genes(cite, n_top_genes=2000, flavor="seurat_v3", batch_key="batch", subset=True)
    scvi.model.TOTALVI.setup_anndata(cite, protein_expression_obsm_key="protein_expression", batch_key="batch")
    tvi = scvi.model.TOTALVI(cite)
    tvi.train(max_epochs=100)
    cite.obsm["X_totalVI"] = tvi.get_latent_representation()
    _, prot = tvi.get_normalized_expression(n_samples=10, return_mean=True)
    sc.pp.neighbors(cite, use_rep="X_totalVI"); sc.tl.umap(cite)
    cite.obs[[f"prot_{p}" for p in prot.columns[:4]]] = prot.iloc[:, :4].values
    sc.pl.umap(cite, color=["batch"] + [f"prot_{p}" for p in prot.columns[:4]], ncols=3)

# %% [markdown]
# ## 8. Ligações pico–gene (*peak-to-gene links*)
#
# Pergunta: **qual região regulatória controla qual gene?** No Multiome podemos correlacionar,
# *nas mesmas células*, a acessibilidade de um pico com a expressão de genes próximos.
#
# Problema: células individuais são esparsas demais (a maioria das entradas é 0).
# Solução (ArchR, Signac, SCENIC+): agregar em **metacélulas**, ou seja, grupos pequenos de células muito similares.
#
# Procedimento:
# 1. metacélulas = clusters finos no grafo WNN (~`N_METACELULAS` grupos);
# 2. somar contagens por metacélula → log-CPM;
# 3. para cada gene, picos a ≤ `JANELA` pb do corpo do gene → correlação de Pearson;
# 4. **nulo empírico**: correlação do gene com picos de **outros cromossomos** (sem relação *cis*).

# %%
res_meta = 5.0
sc.tl.leiden(mdata, neighbors_key="wnn", key_added="metacelula", resolution=res_meta, random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
while mdata.obs["metacelula"].nunique() < N_METACELULAS * 0.7 and res_meta < 40:
    res_meta *= 1.6
    sc.tl.leiden(mdata, neighbors_key="wnn", key_added="metacelula", resolution=res_meta, random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
print(f"{mdata.obs['metacelula'].nunique()} metacélulas (resolução {res_meta:.1f})")

def pseudobulk_logcpm(adata, grupos, layer="counts"):
    X = adata.layers[layer]
    G = pd.get_dummies(grupos).astype(np.float32)
    S = (G.values.T @ X) if not hasattr(X, "tocsr") else np.asarray((X.T @ G.values).T)
    S = np.asarray(S)
    cpm = S / S.sum(1, keepdims=True) * 1e6
    return pd.DataFrame(np.log1p(cpm), index=G.columns.astype(str), columns=adata.var_names)

meta = mdata.obs["metacelula"].astype(str).values
E = pseudobulk_logcpm(rna, meta)          # metacélulas × genes
A = pseudobulk_logcpm(atac, meta)         # metacélulas × picos
meta_tipo = mdata.obs.groupby("metacelula", observed=True)["cell_type"].agg(lambda x: x.value_counts().index[0])
E.shape, A.shape

# %%
def parse_intervalo(s):
    c, r = s.split(":"); a, b = r.split("-"); return c, int(a), int(b)

pk = pd.DataFrame([parse_intervalo(p) for p in A.columns], columns=["chrom", "start", "end"], index=A.columns)
gene_iv = rna.var["interval"].astype(str) if "interval" in rna.var else None
genes_alvo = [g for g in rna.var_names[rna.var["highly_variable"]] if gene_iv is not None
              and ":" in gene_iv.get(g, "NA")]
print(f"{len(genes_alvo)} genes (HVG com coordenadas)")

def zscore(M):
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(0)) / (M.std(0) + 1e-9)

Az = zscore(A.values); Ez = pd.DataFrame(zscore(E[genes_alvo].values), columns=genes_alvo)
rng = np.random.default_rng(SEED)
links, nulo = [], []
for g in genes_alvo:
    c, s, e = parse_intervalo(gene_iv[g])
    m = ((pk["chrom"] == c) & (pk["end"] > s - JANELA) & (pk["start"] < e + JANELA)).values
    if m.sum() == 0:
        continue
    r = (Az[:, m] * Ez[g].values[:, None]).mean(0)
    for p, rv in zip(A.columns[m], r):
        mid = (pk.loc[p, "start"] + pk.loc[p, "end"]) // 2
        dist = 0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
        links.append((g, p, rv, dist))
    outros = np.flatnonzero((pk["chrom"] != c).values)
    if len(outros):
        amostra = rng.choice(outros, size=min(20, len(outros)), replace=False)
        nulo.extend((Az[:, amostra] * Ez[g].values[:, None]).mean(0))

links = pd.DataFrame(links, columns=["gene", "pico", "r", "distancia"])
nulo = np.sort(np.asarray(nulo))
from scipy.stats import false_discovery_control
n_maior = len(nulo) - np.searchsorted(nulo, links["r"].values, side="left")   # nº de nulos ≥ r
links["p_emp"] = (n_maior + 1) / (len(nulo) + 1)
links["fdr"] = false_discovery_control(links["p_emp"].values, method="bh")
sig = links[(links["r"] > 0.3) & (links["fdr"] < 0.1)].sort_values("r", ascending=False)
print(f"{len(sig)} ligações significativas entre {sig['gene'].nunique()} genes")
sig.head(15)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(nulo, 60, alpha=.6, density=True, label="nulo (trans)")
ax[0].hist(links["r"], 60, alpha=.6, density=True, label="cis (±janela)")
ax[0].set_xlabel("r de Pearson"); ax[0].legend()
ax[1].scatter(sig["distancia"] / 1e3, sig["r"], s=6)
ax[1].set_xlabel("distância ao gene (kb)"); ax[1].set_ylabel("r")
plt.tight_layout(); plt.show()

# %%
# Exemplo: gene com mais ligações → acessibilidade × expressão nas metacélulas
if len(sig):
    g_top = sig["gene"].value_counts().index[0]
    p_top = sig[sig["gene"] == g_top].iloc[0]["pico"]
    cores = pd.Categorical(meta_tipo.reindex(E.index).astype(str))
    plt.figure(figsize=(4.5, 3.5))
    for k, cat in enumerate(cores.categories):
        sel = cores == cat
        plt.scatter(A.loc[sel, p_top], E.loc[sel, g_top], s=14, label=cat)
    plt.xlabel(f"acessibilidade {p_top}"); plt.ylabel(f"expressão {g_top}")
    plt.legend(fontsize=6, bbox_to_anchor=(1, 1)); plt.title("Ligação pico–gene (metacélulas)"); plt.show()

# %% [markdown]
# ## 9. Ponte com o fluxo em R (Signac/Seurat)
#
# | Passo | Signac / Seurat (R) | Este notebook (Python) |
# |---|---|---|
# | Ler cellranger-arc | `Read10X_h5()` + `CreateChromatinAssay()` | `mu.read_10x_h5()` |
# | Normalizar ATAC | `RunTFIDF()` | `mu.atac.pp.tfidf()` |
# | Reduzir ATAC | `FindTopFeatures()` + `RunSVD()` (descartar LSI_1) | `mu.atac.tl.lsi()` (descartar LSI_1) |
# | WNN | `FindMultiModalNeighbors()` | `mu.pp.neighbors()` |
# | UMAP WNN | `RunUMAP(nn.name = "weighted.nn")` | `mu.tl.umap(neighbors_key="wnn")` |
# | Pico–gene | `LinkPeaks()` (correção por GC/acessibilidade) | correlação em metacélulas + nulo *trans* |
# | Motivos | `AddMotifs()` + `RunChromVAR()` | `pychromvar` / SnapATAC2 (notebook 02) |
# | Cobertura | `CoveragePlot()` | `snap.ex.export_coverage()` + IGV |
#
# ```r
# # Referência (não executado aqui)
# library(Seurat); library(Signac)
# obj <- FindMultiModalNeighbors(obj, reduction.list = list("pca", "lsi"), dims.list = list(1:30, 2:30))
# obj <- RunUMAP(obj, nn.name = "weighted.nn", reduction.name = "wnn.umap")
# obj <- LinkPeaks(obj, peak.assay = "ATAC", expression.assay = "SCT", genes.use = VariableFeatures(obj))
# ```

# %% [markdown]
# ## 10. Salvar para a Parte B (GRN)

# %%
mdata.update()
for m in ("rna", "atac"):
    for k in list(mdata.mod[m].obsm.keys()):
        if isinstance(mdata.mod[m].obsm[k], pd.DataFrame):
            del mdata.mod[m].obsm[k]
mdata.write(OUT / "aula2_multiome.h5mu", compression="gzip")
links.to_csv(OUT / "aula2_links_pico_gene.csv", index=False)
modelo_mvi.save(OUT / "modelo_multivi", overwrite=True)
print("salvo em", OUT)

try:
    import session_info2; display(session_info2.session_info())   # noqa: F821
except ImportError:
    sc.logging.print_versions()

# %% [markdown]
# ## 🏋️ Exercícios
# 1. Compare o Leiden no grafo WNN com a clusterização **multiplex** do muon
   (`mu.tl.leiden(mdata, mod_weights={"rna": .5, "atac": .5})`, que usa os grafos de cada modalidade). Algum tipo celular se funde?
# 2. **MultiVI com dados não pareados:** esconda o ATAC de metade das células e o RNA da outra metade
#    (`scvi.data.organize_multiome_anndatas`). O espaço latente ainda separa os tipos celulares?
# 3. Mude `JANELA` para 50 kb e 250 kb. Como muda o número de ligações? E a distribuição das distâncias?
# 4. Escolha um gene marcador (ex.: `CD14`, `MS4A1`) e liste os picos ligados a ele. Algum fica em um íntron?
