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
# # Bônus: seus próprios dados, do output do Cell Ranger à análise
#
# **Curso:** Análise Avançada de Dados de Single-cell RNA-seq, ATAC-seq e Multiome (10x Genomics) no Google Colab
# **Ministrante:** Dra. Cristal Villalba · [github.com/kur1sutaru](https://github.com/kur1sutaru)
#
# ---
#
# Este notebook parte do que **você recebe da *core facility***: a pasta `outs/` do `cellranger count`.
#
# | Arquivo | Para quê |
# |---|---|
# | `metrics_summary.csv` / `web_summary.html` | o laudo de qualidade da corrida |
# | `raw_feature_bc_matrix.h5` | **todos** os barcodes, incluindo gotículas vazias (sopa de RNA ambiente) |
# | `filtered_feature_bc_matrix.h5` | barcodes que o Cell Ranger chamou de células |
#
# **Dados de exemplo:** epitélio de língua de camundongo C57BL/6 (macho), Cell Ranger 9.0.1, GRCm39-2024-A, 3' v3,
# GEO [**GSE293526**](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293526), de
# Matsu-ura T. *et al.* Circadian clock–gated cell renewal controls time-dependent changes in taste sensitivity.
# *PNAS* 122(19):e2421421122 (2025). [doi:10.1073/pnas.2421421122](https://doi.org/10.1073/pnas.2421421122)
#
# **Para usar os seus dados:** mude `USAR_EXEMPLO = False` e faça upload da sua pasta `outs/` (ou monte o Drive).

# %% [markdown]
# ## 0. Instalação 🔧

# %% tags=["colab-only"]
# %pip install -q scanpy==1.12.4 anndata==0.13.4 scvi-tools==1.5.1 scikit-learn==1.7.2 leidenalg==0.12.0 session-info2

# %%
import os, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
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
USAR_EXEMPLO = True
PASTA_OUTS = "/content/drive/MyDrive/meu_experimento/outs"     # usada se USAR_EXEMPLO = False
REPO_RAW = "https://github.com/kur1sutaru/curso-singlecell-multiome-avancado/raw/main/data/exemplo_GSE293526"
MAX_EPOCHS_SCAR = 200
LIMIAR_MANTIDO = 0.6                   # se o scAR mantiver < 60% das UMIs (mediana), não usamos a camada descontaminada
ESPECIE_MT = "mt-"                     # camundongo: "mt-"; humano: "MT-"
random.seed(SEED); np.random.seed(SEED)

import scvi
scvi.settings.seed = SEED

# %% [markdown]
# ## 1. Obter os arquivos 📥

# %% tags=["load"]
import urllib.request
if USAR_EXEMPLO:
    OUTS = DATA / "exemplo_GSE293526"; OUTS.mkdir(parents=True, exist_ok=True)
    for f in ["metrics_summary.csv", "raw_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"]:
        if not (OUTS / f).exists():
            print("baixando", f); urllib.request.urlretrieve(f"{REPO_RAW}/{f}", OUTS / f)
else:
    OUTS = Path(PASTA_OUTS)
sorted(p.name for p in OUTS.iterdir())

# %% [markdown]
# ## 2. Lendo o laudo: `metrics_summary.csv`
# Valores de referência (guia técnico da 10x para 3' v3; variam por tecido):
#
# | Métrica | Atenção se… | O que sugere |
# |---|---|---|
# | Fraction Reads in Cells | < 70% | muito RNA ambiente (lise, tecido difícil de dissociar) |
# | Reads Mapped Confidently to Transcriptome | < 30% | referência errada, contaminação, muitos íntrons não contados |
# | Sequencing Saturation | baixa (< 30%) | ainda vale sequenciar mais |
# | Q30 Bases in Barcode / UMI | < 80% / < 65% | problema na corrida de sequenciamento |
# | Valid Barcodes | < 75% | problema de química/biblioteca |

# %%
m = pd.read_csv(OUTS / "metrics_summary.csv").T.rename(columns={0: "valor"})

def num(v):
    v = str(v).replace(",", "")
    return float(v.rstrip("%")) / 100 if v.endswith("%") else float(v)

m["num"] = m["valor"].map(num)
alertas = {"Fraction Reads in Cells": ("<", 0.70), "Reads Mapped Confidently to Transcriptome": ("<", 0.30),
           "Valid Barcodes": ("<", 0.75), "Q30 Bases in Barcode": ("<", 0.80), "Q30 Bases in UMI": ("<", 0.65)}
m["status"] = [("⚠️" if (k in alertas and m.loc[k, "num"] < alertas[k][1]) else "✅") if k in alertas else ""
               for k in m.index]
m[["valor", "status"]]

# %% [markdown]
# ## 3. Matriz *raw*: células × gotículas vazias
# ### 3.1 *Knee plot* (barcode rank)
# Ordenamos barcodes por nº de UMIs. O "joelho" separa gotículas com célula do platô de **gotículas vazias**,
# que contêm só RNA ambiente.

# %%
raw = sc.read_10x_h5(OUTS / "raw_feature_bc_matrix.h5"); raw.var_names_make_unique()
filt = sc.read_10x_h5(OUTS / "filtered_feature_bc_matrix.h5"); filt.var_names_make_unique()
umis = np.asarray(raw.X.sum(1)).ravel()
ordem = np.argsort(-umis)
eh_celula = raw.obs_names[ordem].isin(filt.obs_names)
rank = np.arange(1, len(umis) + 1)

plt.figure(figsize=(5, 3.6))
plt.loglog(rank[~eh_celula], umis[ordem][~eh_celula] + 1, ".", ms=1, color="#bbb", label="não-célula")
plt.loglog(rank[eh_celula], umis[ordem][eh_celula] + 1, ".", ms=1, color="#d95f02", label="célula (Cell Ranger)")
plt.xlabel("rank do barcode"); plt.ylabel("UMIs + 1"); plt.legend(markerscale=8); plt.title("Knee plot")
plt.show()
print(f"{raw.n_obs:,} barcodes no raw | {filt.n_obs:,} células chamadas")

# %% [markdown]
# ### 3.2 Qual é a "sopa"? O perfil de RNA ambiente
# Gotículas com poucas UMIs (aqui, 10 a 100) são quase só RNA ambiente. Os genes mais abundantes nelas são
# os que vão **contaminar todas as células**, muitas vezes marcadores do tipo celular mais abundante ou mais frágil.

# %%
vazias = raw[(umis >= 10) & (umis <= 100) & ~raw.obs_names.isin(filt.obs_names)]
perfil = pd.Series(np.asarray(vazias.X.sum(0)).ravel(), index=raw.var_names)
perfil = (perfil / perfil.sum()).sort_values(ascending=False)
print(f"{vazias.n_obs:,} gotículas vazias usadas")
perfil.head(20).rename("fração da sopa").to_frame()

# %% [markdown]
# ## 4. Remover RNA ambiente com **scAR** (scvi-tools), e **checar se não removeu demais**
# O **scAR** (Sheng et al., *Nat Commun* 2024; disponível no scvi-tools) é um VAE que modela cada contagem como
# *sinal nativo + ruído ambiente*. O ruído segue o perfil das gotículas vazias.
# Alternativas: **CellBender** (também usa o raw; GPU), **SoupX** e **DecontX** (R).
#
# ⚠️ **Armadilha:** neste tecido, o RNA ambiente é dominado por **Krt13, Krt14, Krt4**, que são marcadores do
# próprio epitélio (a população dominante). Quando a "sopa" se parece com as células, qualquer descontaminação
# pode **apagar sinal verdadeiro**. Por isso, depois de rodar, medimos quanto foi removido e decidimos com dados.

# %%
adata = filt.copy()
sc.pp.filter_genes(adata, min_cells=3)                   # menos genes → treino mais rápido
adata.layers["counts"] = adata.X.copy()
raw_util = raw[umis >= 2, adata.var_names].copy()        # barcodes com ≥ 2 UMIs (economiza memória)
scvi.external.SCAR.setup_anndata(adata, layer="counts")
scvi.external.SCAR.get_ambient_profile(adata, raw_adata=raw_util, prob=0.995, n_batch=20)
scar = scvi.external.SCAR(adata, ambient_profile="ambient_profile")
scar.train(max_epochs=MAX_EPOCHS_SCAR, early_stopping=True, accelerator="auto")
adata.layers["denoised"] = scar.get_denoised_counts()
del raw_util

# %% [markdown]
# ### 4.1 Diagnóstico da descontaminação
# O que procurar:
# - **fração mantida** muito baixa (< 50–60%) → provável remoção de sinal real;
# - genes da sopa que **somem** de células que deveriam expressá-los (ex.: *Krt13* em suprabasais);
# - genes **ausentes** da sopa que passam a aparecer em mais células (ex.: *Cd74*, *Col1a1*) → o modelo está
#   **inventando** contagens (treino insuficiente ou modelo inadequado ao tecido).

# %%
import scipy.sparse as sp
cnt = adata.layers["counts"]; den = adata.layers["denoised"]
den = den if sp.issparse(den) else sp.csr_matrix(den)
mantido = np.asarray(den.sum(1)).ravel() / np.asarray(cnt.sum(1)).ravel()
print(f"fração de UMIs mantida por célula: mediana {np.median(mantido):.2f} "
      f"(5%–95%: {np.quantile(mantido, .05):.2f}–{np.quantile(mantido, .95):.2f})")

amb = pd.Series(np.asarray(adata.varm["ambient_profile"]).ravel(), index=adata.var_names).sort_values(ascending=False)
genes_check = list(amb.index[:8]) + [g for g in ["Krt5", "Mki67", "Col1a1", "Ptprc", "Cd74"] if g in adata.var_names]
idx = adata.var_names.get_indexer(genes_check)
diag = pd.DataFrame({
    "% do perfil ambiente": (100 * amb.reindex(genes_check)).round(2).values,
    "células >0 antes": np.asarray((cnt[:, idx] > 0).mean(0)).ravel().round(2),
    "células >0 depois": np.asarray((den[:, idx] > 0).mean(0)).ravel().round(2),
    "% UMIs mantidas": (100 * np.asarray(den[:, idx].sum(0)).ravel()
                        / np.maximum(np.asarray(cnt[:, idx].sum(0)).ravel(), 1)).round(1),
}, index=genes_check)
plt.figure(figsize=(4, 2.8)); plt.hist(mantido, 40); plt.xlabel("fração de UMIs mantida"); plt.ylabel("células")
plt.title("Quanto o scAR removeu?"); plt.show()
diag

# %%
# Regra de decisão explícita (e registrada): se a descontaminação removeu a maior parte das UMIs,
# provavelmente está apagando o sinal do tipo celular dominante → seguimos com as contagens cruas.
CAMADA = "denoised" if np.median(mantido) >= LIMIAR_MANTIDO else "counts"
print(f"camada usada na clusterização: '{CAMADA}'  (limiar de fração mantida = {LIMIAR_MANTIDO})")

# %% [markdown]
# ## 5. QC adaptativo (MAD) e doublets

# %%
adata.var["mt"] = adata.var_names.str.startswith(ESPECIE_MT)
adata.var["ribo"] = adata.var_names.str.match(r"^(Rps|Rpl|RPS|RPL)")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], percent_top=None, log1p=True, inplace=True,
                           layer="counts")

def mad_outlier(x, n=5):
    d = np.median(np.abs(x - np.median(x)))
    return (x < np.median(x) - n * d) | (x > np.median(x) + n * d)

adata.obs["outlier"] = (mad_outlier(adata.obs["log1p_total_counts"]) | mad_outlier(adata.obs["log1p_n_genes_by_counts"])
                        | mad_outlier(adata.obs["pct_counts_mt"], 3))
sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")
print(f"{adata.obs['outlier'].sum()} outliers de {adata.n_obs}")

# %%
# Scrublet espera contagens cruas em .X: rodamos numa cópia leve
tmp = sc.AnnData(adata.layers["counts"].copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
sc.pp.scrublet(tmp, random_state=SEED, verbose=False)
adata.obs[["doublet_score", "predicted_doublet"]] = tmp.obs[["doublet_score", "predicted_doublet"]]
print(f"{adata.obs['predicted_doublet'].sum()} doublets previstos")
adata = adata[~adata.obs["outlier"] & ~adata.obs["predicted_doublet"].astype(bool)].copy()
sc.pp.filter_genes(adata, min_cells=5)
adata

# %% [markdown]
# ## 6. Clusterização (na camada escolhida acima)

# %%
adata.X = adata.layers[CAMADA].copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", layer="counts")
sc.pp.pca(adata, n_comps=30, mask_var="highly_variable", random_state=SEED)
sc.pp.neighbors(adata, random_state=SEED); sc.tl.umap(adata, random_state=SEED)
sc.tl.leiden(adata, resolution=0.8, random_state=SEED, flavor="igraph", n_iterations=2, directed=False)
sc.pl.umap(adata, color=["leiden", "log1p_total_counts", "pct_counts_mt"], wspace=0.4)

# %% [markdown]
# ## 7. Anotação por marcadores (epitélio lingual de camundongo)

# %%
MARC = {
    "Basal / progenitor": ["Krt5", "Krt14", "Krt15", "Trp63"],
    "Proliferativo": ["Mki67", "Top2a", "Cdk1"],
    "Suprabasal": ["Krt13", "Krt4", "Krt6a"],
    "Papila filiforme / queratinizado": ["Krt1", "Krt10", "Lor", "Krt36", "Krt84"],
    "Gustativa tipo II": ["Trpm5", "Gnat3", "Plcb2", "Tas1r3"],
    "Gustativa tipo III": ["Car4", "Pkd2l1", "Snap25"],
    "Fibroblasto": ["Col1a1", "Pdgfra", "Dcn"],
    "Imune": ["Ptprc", "Cd74", "Lyz2"],
    "Endotélio": ["Pecam1", "Cdh5"],
    "Músculo": ["Acta1", "Myh4", "Ckm"],
}
MARC = {k: [g for g in v if g in adata.var_names] for k, v in MARC.items()}
MARC = {k: v for k, v in MARC.items() if v}
sc.pl.dotplot(adata, MARC, groupby="leiden", standard_scale="var")

# %%
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
sc.get.rank_genes_groups_df(adata, None).groupby("group").head(5).pivot_table(
    index="group", values="names", aggfunc=lambda x: ", ".join(x))

# %% [markdown]
# ### Conexão com o artigo: genes do relógio circadiano
# O estudo mostra que a renovação celular do epitélio gustativo é **controlada pelo relógio circadiano**.
# Em quais populações os genes do relógio são expressos?

# %%
relogio = [g for g in ["Arntl", "Clock", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Nr1d1", "Nr1d2", "Dbp", "Tef"]
           if g in adata.var_names]
sc.pl.dotplot(adata, relogio, groupby="leiden", standard_scale="var", title="Genes do relógio")

# %%
adata.write_h5ad(OUT / "bonus_cellranger_processado.h5ad", compression="gzip")
try:
    import session_info2; display(session_info2.session_info())   # noqa: F821
except ImportError:
    sc.logging.print_versions()

# %% [markdown]
# ## 🏋️ Exercícios
# 1. Force `CAMADA = "denoised"` e `CAMADA = "counts"` e compare. Algum cluster "some" ou "aparece"? Quais marcadores mudam?
   Com o seu conhecimento do tecido, qual resultado é mais plausível?
# 2. O Cell Ranger 9 chama células com um algoritmo tipo EmptyDrops. Encontre, no *knee plot*, barcodes com muitas UMIs
#    que **não** foram chamados de células. O que eles expressam?
# 3. Rode o notebook 01 inserindo esta amostra como um terceiro "lote". Por que integrar camundongo com humano não faz sentido
#    sem mapear ortólogos?
