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
# # Aula 2 · Parte C: Reprodutibilidade e escalabilidade
# ## Do notebook ao pipeline, e do pipeline à publicação (FAIR)
#
# **Curso:** Análise Avançada de Dados de Single-cell RNA-seq, ATAC-seq e Multiome (10x Genomics) no Google Colab
# **Ministrante:** Dra. Cristal Villalba · [github.com/kur1sutaru](https://github.com/kur1sutaru)
#
# ---
#
# > "Funciona na minha máquina" não é um resultado científico.
#
# | Camada | Problema | Ferramenta |
# |---|---|---|
# | Ambiente | versões mudam e quebram código (vimos 4 exemplos neste curso!) | versões fixadas, `pip freeze`, `session-info2`, containers |
# | Aleatoriedade | UMAP, Leiden e redes neurais dependem de sementes | `SEED` único, determinismo do PyTorch |
# | Parâmetros | valores "mágicos" espalhados no código | célula `parameters` + **papermill** |
# | Dados | arquivo mudou? download incompleto? | checksums (MD5/SHA256), manifesto |
# | Orquestração | rodar 30 amostras à mão | **Snakemake** / **Nextflow** |
# | Publicação | dados e código achados, acessíveis, interoperáveis e reutilizáveis | GEO, Zenodo, CELLxGENE, GitHub + DOI, `CITATION.cff` |

# %% [markdown]
# ## 0. Instalação 🔧

# %% tags=["colab-only"]
# %pip install -q scanpy==1.12.4 anndata==0.13.4 snakemake==9.27.0 papermill session-info2 leidenalg==0.12.0

# %%
import os, sys, json, hashlib, platform, subprocess, random
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

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
ENTRADA_LOCAL = ""      # .h5ad com contagens cruas para o pipeline (vazio = baixar o pbmc3k da 10x)

# %% [markdown]
# ## 1. Ambiente: registre **tudo** o que roda
# ### 1.1 Lições deste curso
# Cada um destes problemas apareceu ao preparar as aulas, com versões **atuais** (out/2026) dos pacotes:
#
# | Pacote | Problema | Causa |
# |---|---|---|
# | `scanpy.external.pp.harmony_integrate` | erro de dimensão | `harmonypy 2.0` mudou a orientação de `Z_corr` |
# | `celltypist.train` | `TypeError: multi_class` | removido do `LogisticRegression` no scikit-learn 1.8 |
# | `episcanpy` | não importa | usa `np.NaN` (numpy 2) e `anndata.read` (anndata ≥ 0.11) |
# | `pychromvar.compute_deviations` | `TypeError: dtype` | `AnnData(dtype=)` removido no anndata 0.11 |
#
# **Moral:** fixe versões que funcionam *juntas*, registre-as junto com os resultados e teste antes de atualizar.

# %%
def registrar_ambiente(destino: Path) -> dict:
    info = {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "colab": IN_COLAB,
        "seed": SEED,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (destino / "requirements-lock.txt").write_text(freeze)
    (destino / "ambiente.json").write_text(json.dumps(info, indent=2))
    return info

registrar_ambiente(OUT)

# %%
try:
    import session_info2
    display(session_info2.session_info())   # noqa: F821
except ImportError:
    sc.logging.print_versions()

# %% [markdown]
# ## 2. Aleatoriedade sob controle
# Leiden e UMAP usam números aleatórios. **Mesma semente → mesmo resultado** (no mesmo ambiente).
# Semente diferente → partições ligeiramente diferentes, o que é bom para testar **robustez**.

# %%
def fixar_sementes(seed=0):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

adata = sc.datasets.pbmc68k_reduced()          # vem dentro do Scanpy (sem download)
from sklearn.metrics import adjusted_rand_score

def clusterizar(seed):
    a = adata.copy()
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.leiden(a, resolution=1.0, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    return a.obs["leiden"].values

c0, c0b, c1 = clusterizar(0), clusterizar(0), clusterizar(1)
print(f"ARI(semente 0, semente 0) = {adjusted_rand_score(c0, c0b):.3f}")
print(f"ARI(semente 0, semente 1) = {adjusted_rand_score(c0, c1):.3f}")

# %% [markdown]
# > No GPU, algumas operações do PyTorch **não são determinísticas** mesmo com semente (somas atômicas).
# > Para modelos do scvi-tools, reporte a semente, o número de épocas e a versão. Para resultados críticos,
# > treine com 3 a 5 sementes e mostre que a conclusão não muda.

# %% [markdown]
# ## 3. Parâmetros explícitos e execução em lote com **papermill**
# Todos os notebooks do curso têm uma célula marcada com a *tag* `parameters`. O **papermill** executa um notebook
# substituindo esses valores e salva uma cópia **executada**, que vira o registro do que foi rodado.
#
# ```bash
# papermill 01_scRNA_doublets_integracao_anotacao.ipynb saida_seed1.ipynb -p SEED 1 -p MAX_EPOCHS_SCVI 400
# ```
# Abaixo, um exemplo mínimo que criamos e executamos com três sementes.

# %%
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

mini = new_notebook(cells=[
    new_code_cell("SEED = 0\nRESOLUCAO = 1.0", metadata={"tags": ["parameters"]}),
    new_code_cell(
        "import scanpy as sc\n"
        "a = sc.datasets.pbmc68k_reduced()\n"
        "sc.pp.neighbors(a, random_state=SEED)\n"
        "sc.tl.leiden(a, resolution=RESOLUCAO, random_state=SEED, flavor='igraph', n_iterations=2, directed=False)\n"
        "n_clusters = a.obs['leiden'].nunique()\n"
        "print(SEED, RESOLUCAO, n_clusters)"),
])
mini.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
(OUT / "papermill").mkdir(exist_ok=True)
nbformat.write(mini, OUT / "papermill" / "mini.ipynb")

import papermill as pm
for s in (0, 1, 2):
    pm.execute_notebook(OUT / "papermill" / "mini.ipynb", OUT / "papermill" / f"mini_seed{s}.ipynb",
                        parameters={"SEED": s, "RESOLUCAO": 0.8}, progress_bar=False, kernel_name="python3")
sorted(p.name for p in (OUT / "papermill").glob("mini_seed*.ipynb"))

# %% [markdown]
# ## 4. Dados: checksums e manifesto
# Um *checksum* garante que o arquivo que **você** analisou é o mesmo que **outra pessoa** baixou.
# A 10x, o GEO e o Zenodo publicam MD5 dos arquivos: confira antes de analisar.

# %%
def md5(path, bloco=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bloco), b""):
            h.update(chunk)
    return h.hexdigest()

arquivos = [p for p in list(DATA.rglob("*.h5")) + list(DATA.rglob("*.h5ad")) + list(OUT.glob("*.h5*"))
            if p.is_file()][:20]
manifesto = pd.DataFrame([{"arquivo": str(p), "bytes": p.stat().st_size, "md5": md5(p)} for p in arquivos])
manifesto.to_csv(OUT / "manifesto_dados.csv", index=False)
manifesto

# %% [markdown]
# ### Persistência no Colab
# O disco do Colab é **apagado** quando a sessão termina. Para guardar resultados entre aulas:
# ```python
# from google.colab import drive
# drive.mount("/content/drive")
# OUT = Path("/content/drive/MyDrive/curso_singlecell/resultados")
# ```

# %% [markdown]
# ## 5. Do notebook ao pipeline: **Snakemake**
#
# Notebooks são ótimos para **explorar**; pipelines são melhores para **produzir**:
# - cada regra declara **entradas → saídas**, e o Snakemake monta o grafo (DAG);
# - só refaz o que mudou (como o `make`);
# - roda em paralelo, em cluster (SLURM) ou na nuvem, com ambientes conda/containers por regra.
#
# Vamos escrever um pipeline mínimo de scRNA-seq: `baixar → qc → normalizar → cluster (× resoluções) → marcadores → relatório`.

# %%
PIPE = OUT / "pipeline_snakemake"
(PIPE / "scripts").mkdir(parents=True, exist_ok=True)

(PIPE / "config.yaml").write_text(f"""\
amostra: pbmc3k
resolucoes: [0.5, 1.0]
qc:
  min_genes: 200
  max_pct_mt: 15
n_hvg: 2000
seed: 0
entrada_local: "{ENTRADA_LOCAL}"     # caminho para um .h5ad local; vazio = baixar pbmc3k
""")

(PIPE / "Snakefile").write_text('''\
configfile: "config.yaml"

A = config["amostra"]
RES = config["resolucoes"]

rule all:
    input:
        f"resultados/{A}/relatorio.csv"

rule baixar:
    output: f"dados/{A}_bruto.h5ad"
    params: entrada=config["entrada_local"]
    script: "scripts/baixar.py"

rule qc:
    input: rules.baixar.output
    output: "resultados/{amostra}/qc.h5ad"
    params: min_genes=config["qc"]["min_genes"], max_pct_mt=config["qc"]["max_pct_mt"]
    log: "logs/{amostra}_qc.log"
    script: "scripts/qc.py"

rule normalizar:
    input: "resultados/{amostra}/qc.h5ad"
    output: "resultados/{amostra}/norm.h5ad"
    params: n_hvg=config["n_hvg"], seed=config["seed"]
    script: "scripts/normalizar.py"

rule cluster:
    input: "resultados/{amostra}/norm.h5ad"
    output: "resultados/{amostra}/cluster_res{res}.csv"
    params: seed=config["seed"]
    script: "scripts/cluster.py"

rule marcadores:
    input: norm="resultados/{amostra}/norm.h5ad", cl="resultados/{amostra}/cluster_res{res}.csv"
    output: "resultados/{amostra}/marcadores_res{res}.csv"
    script: "scripts/marcadores.py"

rule relatorio:
    input:
        cl=expand("resultados/{{amostra}}/cluster_res{res}.csv", res=RES),
        mk=expand("resultados/{{amostra}}/marcadores_res{res}.csv", res=RES)
    output: "resultados/{amostra}/relatorio.csv"
    script: "scripts/relatorio.py"
''')

scripts = {
"baixar.py": '''\
import shutil, scanpy as sc
if snakemake.params.entrada:
    shutil.copy(snakemake.params.entrada, snakemake.output[0])
else:
    sc.datasets.pbmc3k().write_h5ad(snakemake.output[0])
''',
"qc.py": '''\
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
open(snakemake.log[0], "w").write(f"{n0} -> {a.n_obs} células\\n")
''',
"normalizar.py": '''\
import scanpy as sc
a = sc.read_h5ad(snakemake.input[0])
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=snakemake.params.n_hvg, flavor="seurat_v3", layer="counts")
sc.pp.pca(a, n_comps=30, mask_var="highly_variable", random_state=snakemake.params.seed)
sc.pp.neighbors(a, random_state=snakemake.params.seed)
a.write_h5ad(snakemake.output[0])
''',
"cluster.py": '''\
import scanpy as sc
a = sc.read_h5ad(snakemake.input[0])
res = float(snakemake.wildcards.res)
sc.tl.leiden(a, resolution=res, random_state=snakemake.params.seed, flavor="igraph", n_iterations=2, directed=False)
a.obs[["leiden"]].to_csv(snakemake.output[0])
''',
"marcadores.py": '''\
import scanpy as sc, pandas as pd
a = sc.read_h5ad(snakemake.input.norm)
a.obs["leiden"] = pd.read_csv(snakemake.input.cl, index_col=0)["leiden"].astype(str).astype("category").values
sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon")
sc.get.rank_genes_groups_df(a, None).groupby("group").head(5).to_csv(snakemake.output[0], index=False)
''',
"relatorio.py": '''\
import pandas as pd, re
linhas = []
for cl, mk in zip(snakemake.input.cl, snakemake.input.mk):
    res = re.search(r"res([0-9.]+)\\.csv", cl).group(1)
    c = pd.read_csv(cl, index_col=0); m = pd.read_csv(mk)
    top = m.groupby("group")["names"].first().to_dict()
    for k, n in c["leiden"].value_counts().sort_index().items():
        linhas.append({"resolucao": res, "cluster": k, "n_celulas": n, "marcador_top": top.get(str(k), top.get(k))})
pd.DataFrame(linhas).to_csv(snakemake.output[0], index=False)
''',
}
for nome, txt in scripts.items():
    (PIPE / "scripts" / nome).write_text(txt)
sorted(str(p.relative_to(PIPE)) for p in PIPE.rglob("*") if p.is_file())

# %% [markdown]
# ### 5.1 *Dry-run*: o que **seria** executado?

# %%
def snake(*args):
    """Roda o Snakemake na pasta do pipeline e mostra só o final do log."""
    r = subprocess.run([sys.executable, "-m", "snakemake", *args], cwd=PIPE, capture_output=True, text=True)
    log = (r.stdout + r.stderr).strip().splitlines()
    print("\n".join(l for l in log[-25:] if l.strip()))
    if r.returncode != 0:
        raise RuntimeError("Snakemake falhou (veja o log acima)")

snake("-n", "--cores", "1", "--quiet", "rules")

# %% [markdown]
# ### 5.2 O grafo de execução (DAG)

# %%
r = subprocess.run(f"{sys.executable} -m snakemake --dag --cores 1 | dot -Tpng > dag.png",
                   shell=True, cwd=PIPE, capture_output=True, text=True)
if (PIPE / "dag.png").exists() and (PIPE / "dag.png").stat().st_size > 0:
    from IPython.display import Image, display
    display(Image(str(PIPE / "dag.png"), width=420))
else:
    print("graphviz indisponível:", r.stderr[-500:])

# %% [markdown]
# ### 5.3 Executar

# %%
snake("--cores", "2", "--quiet")
pd.read_csv(PIPE / "resultados" / "pbmc3k" / "relatorio.csv").head(20)

# %% [markdown]
# Rode de novo: o Snakemake responde **"Nothing to be done"**. Agora mude `resolucoes` no `config.yaml`
# para `[0.5, 1.0, 2.0]` e rode: **só** a nova resolução é calculada.

# %%
snake("--cores", "2", "--quiet")

# %% [markdown]
# ## 6. Nextflow: visão geral
# O **Nextflow** é a alternativa mais usada em *core facilities* e pela comunidade **nf-core**
# (ex.: [`nf-core/scrnaseq`](https://nf-co.re/scrnaseq), que roda Cell Ranger, STARsolo, alevin-fry e kallisto|bustools).
#
# | | Snakemake | Nextflow |
# |---|---|---|
# | Linguagem | Python | Groovy (DSL2) |
# | Modelo | regras por **arquivos** (puxa do alvo) | **processos** ligados por **canais** (empurra os dados) |
# | Comunidade | Snakemake Workflow Catalog | **nf-core** (pipelines revisados) |
# | Execução | local, SLURM, nuvem, containers | local, SLURM, AWS Batch, Google Batch, containers |
#
# O mesmo passo de QC em Nextflow DSL2:
# ```groovy
# process QC {
#     container 'quay.io/biocontainers/scanpy:1.10.1--pyhdfd78af_0'
#     input:  tuple val(amostra), path(h5ad)
#     output: tuple val(amostra), path("${amostra}_qc.h5ad")
#     script:
#     """
#     qc.py --entrada ${h5ad} --saida ${amostra}_qc.h5ad --min-genes ${params.min_genes}
#     """
# }
# workflow {
#     Channel.fromPath(params.amostras).map { f -> tuple(f.baseName, f) } | QC | NORMALIZAR | CLUSTER
# }
# ```
# Para usar no Colab: `!curl -s https://get.nextflow.io | bash` (requer Java, que já vem instalado).

# %% [markdown]
# ## 7. Publicação FAIR
#
# **F**indable · **A**ccessible · **I**nteroperable · **R**eusable (Wilkinson et al., *Sci Data* 2016)
#
# | O quê | Onde | Checklist |
# |---|---|---|
# | FASTQ / BAM | **GEO** (+ SRA) ou ArrayExpress (+ ENA) | metadados MINSEQE, química 10x, versão do Cell Ranger |
# | Matrizes processadas | GEO (suplementar), **Zenodo** (DOI), **CELLxGENE** | `.h5ad`/`.h5mu` com contagens cruas em `layers["counts"]` e anotações em `obs` |
# | Código | **GitHub** + release arquivado no **Zenodo** (DOI) | `README`, `LICENSE`, `CITATION.cff`, ambiente fixado |
# | Ambiente | `requirements-lock.txt`, `environment.yml`, **Docker/Apptainer** | versão de Python e do CUDA |
# | Parâmetros | `config.yaml`, notebooks executados (papermill) | sementes, limiares de QC, versões de referência (GRCh38-2024-A) |
#
# **Este repositório** segue esse modelo: veja `README.md`, `CITATION.cff`, `environment/` e `pipelines/`.

# %%
checklist = pd.DataFrame({
    "item": ["Versões fixadas", "Semente registrada", "Parâmetros explícitos", "Checksums dos dados",
             "Pipeline (Snakemake/Nextflow)", "Contagens cruas salvas", "Código com licença e DOI",
             "Dados brutos em repositório público"],
    "neste curso": ["✅", "✅", "✅", "✅", "✅", "✅", "✅ (GitHub + Zenodo)", "✅ (10x / GEO)"],
})
checklist

# %% [markdown]
# ## 🏋️ Exercícios
# 1. Adicione ao Snakefile uma regra `umap` por resolução e inclua a figura no `rule all`.
# 2. Adicione `conda: "envs/scanpy.yaml"` às regras e rode com `--software-deployment-method conda`.
# 3. Use o papermill para rodar o notebook 01 com `SEED` 0, 1 e 2 e compare a tabela de métricas de integração.
# 4. Escreva o `CITATION.cff` do seu próximo artigo (modelo neste repositório).
