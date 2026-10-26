# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   colab:
#     provenance: []
# ---

# %% [markdown]
# # Aula 2 · Parte B: Redes regulatórias (GRN) a partir de Multiome
# ## TF → enhancer → gene-alvo: construindo *eRegulons*
#
# **Curso:** Análise Avançada de Dados de Single-cell RNA-seq, ATAC-seq e Multiome (10x Genomics) no Google Colab
# **Ministrante:** Dra. Cristal Villalba · [github.com/kur1sutaru](https://github.com/kur1sutaru)
#
# ---
#
# Um **eRegulon** (SCENIC+, Bravo González-Blas et al., *Nat Methods* 2023) é um trio com três evidências:
#
# ```
#   TF  ──(motivo no enhancer)──►  região  ──(acessibilidade ~ expressão)──►  gene-alvo
#    └────────────────────(expressão do TF ~ expressão do alvo)──────────────────┘
# ```
#
# | Evidência | Dado | Como medimos aqui |
# |---|---|---|
# | o TF **pode** se ligar à região | sequência | motivos CIS-BP (`snap.tl.add_tf_binding`) |
# | a região **regula** o gene | ATAC × RNA | correlação + regressão em metacélulas (`snap.tl.add_cor_scores`, `add_regr_scores`) |
# | o TF **está ativo** | RNA | correlação TF–alvo e atividade do TF (decoupler + CollecTRI) |
#
# ### Por que não rodar o SCENIC+ inteiro aqui?
# O SCENIC+ é a referência, mas exige **Python 3.11**, bancos *cisTarget* de vários GB e muita RAM.
# Ele não roda de forma confiável no Colab gratuito. Este notebook constrói **os mesmos trios** com ferramentas leves
# (SnapATAC2 + decoupler). A **seção 8** traz o roteiro do SCENIC+ real, para Colab Pro ou servidor.

# %% [markdown]
# ## 0. Instalação 🔧

# %% tags=["colab-only"]
# %pip install -q snapatac2==2.10.0 scanpy==1.12.4 anndata==0.13.4 mudata==0.4.1 muon==0.1.9 \
#     decoupler==2.2.0 celltypist==1.7.1 scikit-learn==1.7.2 networkx leidenalg==0.12.0 session-info2

# %% [markdown]
# ## 1. Imports e parâmetros ⚙️

# %%
import os, warnings, random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
import snapatac2 as snap
import decoupler as dc
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
os.environ.setdefault("SNAP_DATA_DIR", str(DATA / "snapatac2_cache"))

# %% tags=["parameters"]
SEED = 0
N_METACELULAS = 200         # metacélulas puras por tipo celular
JANELA = 150_000            # região considerada ao redor de cada gene (pb)
MIN_CELLS_PEAK = 0.02       # picos abertos em ≥ 2% das células
N_GENES = 2000              # genes-alvo candidatos (HVGs)
LIMIAR_R_REGIAO = 0.3       # cor(região, gene) mínima
LIMIAR_R_TF = 0.3           # cor(TF, gene) mínima (eRegulons ativadores)
MIN_ALVOS = 10              # tamanho mínimo de um eRegulon
random.seed(SEED); np.random.seed(SEED)

# %% [markdown]
# ## 2. Dados: checkpoint da Parte A 📥
# Se você rodou o notebook `03_multiome_integracao` nesta sessão, carregamos o `aula2_multiome.h5mu`.
# Caso contrário, esta célula refaz um pré-processamento **mínimo** (download + QC + CellTypist).

# %% tags=["load"]
GENOMA = snap.genome.hg38
CKPT = OUT / "aula2_multiome.h5mu"

if CKPT.exists():
    mdata = mu.read_h5mu(CKPT)
else:
    import urllib.request, celltypist
    from celltypist import models
    url = ("https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/"
           "pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5")
    h5 = DATA / "pbmc3k_multiome_filtered.h5"
    if not h5.exists():
        urllib.request.urlretrieve(url, h5)
    mdata = mu.read_10x_h5(h5, extended=False); mdata.var_names_make_unique()
    for m in ("rna", "atac"):
        sc.pp.calculate_qc_metrics(mdata.mod[m], percent_top=None, log1p=False, inplace=True)
        q = mdata.mod[m].obs["total_counts"].quantile([0.02, 0.98]).values
        mu.pp.filter_obs(mdata.mod[m], "total_counts", lambda x: (x > q[0]) & (x < q[1]))
    mu.pp.intersect_obs(mdata)
    for m in ("rna", "atac"):
        mdata.mod[m].layers["counts"] = mdata.mod[m].X.copy()
    r = mdata.mod["rna"].copy(); sc.pp.normalize_total(r, target_sum=1e4); sc.pp.log1p(r)
    models.download_models(model=["Immune_All_High.pkl"])
    pred = celltypist.annotate(r, model=models.Model.load("Immune_All_High.pkl"))
    mdata.obs["cell_type"] = pred.predicted_labels["predicted_labels"].values
    mdata.mod["rna"].obs["cell_type"] = mdata.obs["cell_type"].values

# %%
rna, atac = mdata.mod["rna"], mdata.mod["atac"]
rna.obs["cell_type"] = mdata.obs["cell_type"].astype(str).values
# remove tipos com poucas células (instáveis para correlação)
contagem = rna.obs["cell_type"].value_counts()
manter = rna.obs["cell_type"].isin(contagem[contagem >= 20].index).values
rna, atac = rna[manter].copy(), atac[manter].copy()
print(rna.shape, atac.shape); contagem

# %% [markdown]
# ## 3. Atividade de TFs a partir do RNA (decoupler + CollecTRI) 📥
# O **CollecTRI** (Müller-Dott et al., *NAR* 2023) reúne ~43 mil interações TF→alvo curadas, com sinal
# (ativação/repressão). A atividade de cada TF por célula é inferida com **ULM** sobre a expressão dos seus alvos.
# Isso é diferente de olhar a expressão do próprio TF: muitos TFs são pouco detectados no scRNA,
# mas seus alvos não.

# %% tags=["load"]
collectri = dc.op.collectri(organism="human")
collectri.head()

# %%
rna.X = rna.layers["counts"].copy()
sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
dc.mt.ulm(rna, collectri, tmin=5)
tf_act = dc.pp.get_obsm(rna, key="score_ulm")
rank_tf = dc.tl.rankby_group(tf_act, groupby="cell_type", reference="rest", method="t-test_overestim_var")
top_tf = rank_tf[rank_tf["stat"] > 0].groupby("group").head(4)
sc.pl.matrixplot(tf_act, var_names=top_tf["name"].unique().tolist(), groupby="cell_type",
                 cmap="RdBu_r", vcenter=0, colorbar_title="atividade (ULM)")

# %% [markdown]
# ## 4. Metacélulas
# Correlações entre região e gene precisam de **agregação** (dados por célula são esparsos demais).
# Usamos metacélulas **puras por tipo celular**: k-means no PCA do RNA, dentro de cada tipo,
# com número de grupos proporcional ao tamanho do tipo (ideia semelhante ao SEACells/MetaCell).

# %%
from sklearn.cluster import KMeans

sc.pp.highly_variable_genes(rna, n_top_genes=N_GENES, flavor="seurat_v3", layer="counts")
sc.pp.pca(rna, n_comps=30, mask_var="highly_variable", random_state=SEED)
meta = np.empty(rna.n_obs, dtype=object)
for ct, idx in rna.obs.groupby("cell_type", observed=True).indices.items():
    k = max(2, int(round(N_METACELULAS * len(idx) / rna.n_obs)))
    k = min(k, max(1, len(idx) // 5))
    lab = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(rna.obsm["X_pca"][idx])
    meta[idx] = [f"{ct}|{l}" for l in lab]
rna.obs["metacelula"] = meta
atac.obs["metacelula"] = meta
print(f"{len(set(meta))} metacélulas; mediana de {pd.Series(meta).value_counts().median():.0f} células cada")

# %%
def pseudobulk(adata, grupos, layer="counts"):
    D = pd.get_dummies(pd.Series(grupos, index=adata.obs_names))
    S = np.asarray((adata.layers[layer].T @ D.values.astype(np.float32)).T)
    S = np.log1p(S / S.sum(1, keepdims=True) * 1e6)
    obs = pd.DataFrame(index=D.columns.astype(str))
    obs["cell_type"] = [c.split("|")[0] for c in obs.index]
    return ad.AnnData(S.astype(np.float32), obs=obs, var=pd.DataFrame(index=adata.var_names))

picos_ok = np.asarray((atac.layers["counts"] > 0).mean(0)).ravel() >= MIN_CELLS_PEAK
gene_mc = pseudobulk(rna[:, rna.var["highly_variable"] | rna.var_names.isin(collectri["source"])], meta)
peak_mc = pseudobulk(atac[:, picos_ok], meta)
print("genes:", gene_mc.shape, "| picos:", peak_mc.shape)

# %% [markdown]
# ## 5. Rede região → gene com SnapATAC2
# 1. `init_network_from_annotation`: liga cada pico a todos os genes cujo TSS está a ≤ `JANELA` pb;
# 2. `add_cor_scores`: correlação de Pearson pico × gene nas metacélulas;
# 3. `add_regr_scores`: regressão *elastic net* (gene ~ todos os seus picos candidatos). Captura a contribuição
#    **condicional** de cada pico, não só a marginal;
# 4. `add_tf_binding`: varre os picos com motivos CIS-BP e adiciona arestas **motivo(TF) → pico**.

# %% tags=["load"]
MOTIVOS_SNAP = snap.datasets.cis_bp(unique=True)

# %%
genes_rede = [g for g in gene_mc.var_names if g in set(rna.var_names[rna.var["highly_variable"]])]
rede = snap.tl.init_network_from_annotation(list(peak_mc.var_names), GENOMA,
                                            upstream=JANELA, downstream=JANELA)
# mantém só genes que estão na matriz de metacélulas
rede = snap.tl.prune_network(rede, node_filter=lambda n: n.type != "gene" or n.id in set(genes_rede))
print(f"rede inicial: {rede.num_nodes()} nós, {rede.num_edges()} arestas")

snap.tl.add_cor_scores(rede, gene_mat=gene_mc, peak_mat=peak_mc)
snap.tl.add_regr_scores(rede, peak_mat=peak_mc, gene_mat=gene_mc)

tfs_expressos = set(gene_mc.var_names) & set(collectri["source"])
motivos = [m for m in MOTIVOS_SNAP if m.name.upper() in {t.upper() for t in tfs_expressos}]
print(f"{len(motivos)} motivos de TFs presentes no RNA")
snap.tl.add_tf_binding(rede, motifs=motivos, genome_fasta=GENOMA)
print(f"rede final: {rede.num_nodes()} nós, {rede.num_edges()} arestas")

# %%
def arestas(rede) -> pd.DataFrame:
    rows = []
    for a, b in rede.edge_list():
        na, nb, d = rede[a], rede[b], rede.get_edge_data(a, b)
        rows.append((na.id, na.type, nb.id, nb.type, d.cor_score, d.regr_score, d.distance))
    return pd.DataFrame(rows, columns=["origem", "tipo_origem", "destino", "tipo_destino",
                                       "cor", "regr", "distancia"])

E = arestas(rede)
regiao_gene = E[(E.tipo_origem == "region") & (E.tipo_destino == "gene")].rename(
    columns={"origem": "regiao", "destino": "gene"})
tf_regiao = E[(E.tipo_origem == "motif") & (E.tipo_destino == "region")].rename(
    columns={"origem": "TF", "destino": "regiao"})[["TF", "regiao"]]
print(f"{len(regiao_gene)} arestas região→gene | {len(tf_regiao)} arestas TF→região")
regiao_gene.sort_values("cor", ascending=False).head()

# %% [markdown]
# ## 6. Montando os trios e os eRegulons
# Trio = (TF, região, gene) com:
# - motivo do TF na região;
# - `cor(região, gene) > LIMIAR_R_REGIAO` e coeficiente de regressão > 0;
# - `cor(TF, gene) > LIMIAR_R_TF` (expressão nas metacélulas): eRegulon **ativador** (`+/+` no jargão do SCENIC+).

# %%
expr = pd.DataFrame(gene_mc.X, index=gene_mc.obs_names, columns=gene_mc.var_names)
z = (expr - expr.mean()) / (expr.std() + 1e-9)
tf_upper = {g.upper(): g for g in expr.columns}

rg = regiao_gene[(regiao_gene["cor"] > LIMIAR_R_REGIAO) & (regiao_gene["regr"].fillna(0) > 0)]
trios = tf_regiao.merge(rg[["regiao", "gene", "cor", "regr", "distancia"]], on="regiao")
trios["TF"] = trios["TF"].map(lambda t: tf_upper.get(t, t))
trios = trios[trios["TF"].isin(expr.columns) & (trios["TF"] != trios["gene"])]

pares = trios[["TF", "gene"]].drop_duplicates()
pares["cor_tf_gene"] = [(z[t] * z[g]).mean() for t, g in zip(pares["TF"], pares["gene"])]
trios = trios.merge(pares, on=["TF", "gene"])
trios = trios[trios["cor_tf_gene"] > LIMIAR_R_TF]
print(f"{len(trios)} trios TF→região→gene | {trios['TF'].nunique()} TFs | {trios['gene'].nunique()} genes")

eregulons = (trios.groupby("TF")
             .agg(n_alvos=("gene", "nunique"), n_regioes=("regiao", "nunique"),
                  alvos=("gene", lambda x: sorted(set(x))))
             .query("n_alvos >= @MIN_ALVOS").sort_values("n_alvos", ascending=False))
eregulons.head(15)

# %% [markdown]
# ### 6.1 Validação externa: os alvos previstos estão no CollecTRI?
# CollecTRI é independente dos nossos dados. Se os eRegulons fazem sentido, devem estar **enriquecidos** em
# interações já conhecidas (teste hipergeométrico por TF).

# %%
from scipy.stats import hypergeom
universo = set(gene_mc.var_names)
conhecidos = collectri[collectri["weight"] > 0].groupby("source")["target"].apply(set)
linhas = []
for tf, row in eregulons.iterrows():
    if tf not in conhecidos.index:
        continue
    K = len(conhecidos[tf] & universo); n = row["n_alvos"]; k = len(set(row["alvos"]) & conhecidos[tf])
    p = hypergeom.sf(k - 1, len(universo), K, n) if K else 1.0
    linhas.append((tf, n, K, k, p))
valid = pd.DataFrame(linhas, columns=["TF", "alvos_previstos", "alvos_CollecTRI", "sobreposicao", "p"])
valid.sort_values("p").head(10)

# %% [markdown]
# ## 7. Atividade dos eRegulons por célula e priorização de alvos
# Como no SCENIC+, pontuamos cada célula pela expressão do **conjunto de alvos** de cada eRegulon (AUCell).

# %%
net_ereg = pd.DataFrame([(f"{tf}_eReg", g) for tf, r in eregulons.iterrows() for g in r["alvos"]],
                        columns=["source", "target"])
if len(net_ereg):
    dc.mt.aucell(rna, net_ereg, tmin=MIN_ALVOS)
    auc = dc.pp.get_obsm(rna, key="score_aucell")
    rk = dc.tl.rankby_group(auc, groupby="cell_type", reference="rest", method="t-test_overestim_var")
    top = rk[rk["stat"] > 0].groupby("group").head(3)["name"].unique().tolist()
    sc.pl.matrixplot(auc, var_names=top, groupby="cell_type", standard_scale="var",
                     cmap="viridis", colorbar_title="AUCell (escalado)")

# %% [markdown]
# ### Priorização para validação experimental
# Um trio é bom candidato a validação (CRISPRi no enhancer, ChIP/CUT&RUN do TF, repórter) quando:
# 1. a ligação região→gene é **forte** (`cor`, `regr`);
# 2. TF e alvo são **coexpressos** (`cor_tf_gene`);
# 3. a região é **específica** de um tipo celular (acessibilidade z-score alta naquele tipo);
# 4. a interação **não é conhecida** (novidade) ou é conhecida (controle positivo).

# %%
acc = pd.DataFrame(peak_mc.X, index=peak_mc.obs_names, columns=peak_mc.var_names)
acc_ct = acc.groupby(peak_mc.obs["cell_type"].values).mean()
acc_z = (acc_ct - acc_ct.mean()) / (acc_ct.std() + 1e-9)
trios["tipo_celular"] = acc_z.idxmax().reindex(trios["regiao"]).values
trios["especificidade"] = acc_z.max().reindex(trios["regiao"]).values
trios["conhecido_CollecTRI"] = [g in conhecidos.get(t, set()) for t, g in zip(trios["TF"], trios["gene"])]
trios["score"] = trios["cor"] * trios["cor_tf_gene"] * np.clip(trios["especificidade"], 0, None)
prior = trios.sort_values("score", ascending=False).drop_duplicates(["TF", "gene"])
prior.groupby("tipo_celular").head(3)[["tipo_celular", "TF", "regiao", "gene", "distancia",
                                       "cor", "cor_tf_gene", "especificidade", "conhecido_CollecTRI", "score"]]

# %%
import networkx as nx
if len(eregulons):
    tfs_plot = eregulons.index[:4].tolist()
    sub = prior[prior["TF"].isin(tfs_plot)].groupby("TF").head(8)
    G = nx.DiGraph()
    for _, r in sub.iterrows():
        G.add_edge(r["TF"], r["gene"], weight=r["score"])
    pos = nx.spring_layout(G, seed=SEED, k=0.8)
    cor_no = ["#d95f02" if n in tfs_plot else "#7570b3" for n in G.nodes]
    plt.figure(figsize=(7, 5.5))
    nx.draw_networkx(G, pos, node_color=cor_no, node_size=650, font_size=7, font_color="white",
                     edge_color="#999", arrowsize=10)
    plt.title("eRegulons top: TF (laranja) → alvos (roxo)"); plt.axis("off"); plt.show()

# %%
trios.to_csv(OUT / "aula2_trios_tf_regiao_gene.csv", index=False)
eregulons.assign(alvos=eregulons["alvos"].map(",".join)).to_csv(OUT / "aula2_eregulons.csv")
prior.head(200).to_csv(OUT / "aula2_alvos_priorizados.csv", index=False)
print("salvo em", OUT)

# %% [markdown]
# ## 8. O SCENIC+ "de verdade" (Colab Pro / servidor) 🧪
#
# O SCENIC+ formaliza exatamente o que fizemos, com mais rigor estatístico:
#
# | Etapa | Este notebook | SCENIC+ |
# |---|---|---|
# | Regiões candidatas | picos MACS3 | picos consenso + **tópicos pycisTopic** + DARs |
# | TF → região | varredura de motivos (CIS-BP) | **pycistarget**: enriquecimento cisTarget + DEM em >30 mil motivos agrupados |
# | Região → gene | correlação + elastic net | **gradient boosting** (GRNBoost2) + correlação de Spearman |
# | TF → gene | correlação nas metacélulas | GRNBoost2 (importância) |
# | eRegulon | trios filtrados | **GSEA** dos alvos ranqueados por TF, com sinal (`+/+`, `+/-`, `-/+`) |
# | Atividade | AUCell (decoupler) | AUCell em genes **e** regiões; eRegulon specificity score (RSS) |
#
# **Como rodar sem brigar com o Colab:** criamos um ambiente **Python 3.11 isolado** com `uv`
# e chamamos o pipeline Snakemake oficial. Rode com `RODAR_SCENICPLUS = True` em um runtime com ≥ 50 GB de RAM.
# > Bancos cisTarget (hg38, SCREEN, *v10 clust*): `https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/screen/mc_v10_clust/region_based/`
# > Anotação de motivos: `https://resources.aertslab.org/cistarget/motif2tf/`.
# > **Confira as versões na [documentação do SCENIC+](https://scenicplus.readthedocs.io)** antes de baixar (vários GB).

# %% tags=["optional"]
RODAR_SCENICPLUS = False

if RODAR_SCENICPLUS:
    import subprocess, textwrap
    ENV = Path("/content/scplus_env")
    cmds = f"""
    set -e
    pip install -q uv
    uv venv -p 3.11 {ENV}
    uv pip install --python {ENV}/bin/python "scenicplus @ git+https://github.com/aertslab/scenicplus@main"
    {ENV}/bin/scenicplus init_snakemake --out_dir /content/scplus_pipeline
    """
    subprocess.run(["bash", "-c", textwrap.dedent(cmds)], check=True)
    print("Edite /content/scplus_pipeline/Snakemake/config/config.yaml (caminhos do cisTopic, AnnData de RNA,")
    print("region sets, bancos cisTarget) e rode:")
    print(f"  cd /content/scplus_pipeline/Snakemake && {ENV}/bin/snakemake --cores 8")

# %% [markdown]
# ## 🏋️ Exercícios
# 1. Aumente `LIMIAR_R_REGIAO` para 0,5. Quantos eRegulons sobrevivem? A sobreposição com o CollecTRI melhora?
# 2. Construa eRegulons **repressores** (`cor_tf_gene < -LIMIAR_R_TF`). Algum TF conhecido como repressor aparece?
# 3. Compare a atividade **ULM/CollecTRI** de um TF com a atividade AUCell do seu eRegulon. Concordam por tipo celular?
# 4. Escolha um trio priorizado e desenhe um experimento de validação (qual técnica? qual controle?).
#
# ---
# ### Referências
# - Bravo González-Blas C. et al. SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. *Nat Methods* 20, 1355–1367 (2023).
# - Zhang K. et al. SnapATAC2: a fast, scalable and versatile tool for analysis of single-cell omics data. *Nat Methods* 21, 217–227 (2024).
# - Müller-Dott S. et al. Expanding the coverage of regulons from high-confidence prior knowledge for accurate estimation of transcription factor activities. *NAR* 51, 10934–10949 (2023).
# - Badia-i-Mompel P. et al. decoupleR: ensemble of computational methods to infer biological activities from omics data. *Bioinform Adv* 2, vbac016 (2022).
