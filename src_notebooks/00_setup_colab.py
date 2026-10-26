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
# # Aula 1 · Parte 0: preparando o Google Colab
#
# **Curso:** Análise Avançada de Dados de Single-cell RNA-seq, ATAC-seq e Multiome (10x Genomics) no Google Colab
# **Ministrante:** Dra. Cristal Villalba · [github.com/kur1sutaru](https://github.com/kur1sutaru)
#
# ---
# Objetivo: em 15 minutos, todos com um ambiente **idêntico** e **funcionando**.
#
# 1. Ligar a GPU: `Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU`
# 2. Rodar este notebook inteiro (`Ambiente de execução → Executar tudo`)
# 3. Conferir a tabela final: tudo ✅

# %% [markdown]
# ## 1. O que o Colab oferece (e os limites)
# | Recurso | Colab gratuito (típico) | Impacto no curso |
# |---|---|---|
# | RAM | ~12 GB | datasets de até ~10 mil células com folga |
# | Disco | ~100 GB, **apagado** ao fim da sessão | salve resultados no Drive |
# | GPU | T4 16 GB (com cota diária) | scVI/MultiVI ~10× mais rápido |
# | Tempo | sessão cai após inatividade / ~12 h | rode as células "pesadas" primeiro |

# %%
import sys, platform, shutil, subprocess, os
print("Python:", sys.version.split()[0], "|", platform.platform())
try:
    import psutil
    print(f"RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")
except ImportError:
    pass
print(f"Disco livre: {shutil.disk_usage('/').free / 1e9:.0f} GB")
gpu = ""
if shutil.which("nvidia-smi"):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "nenhuma (ok para os notebooks 02, 04 e 05)")

# %% [markdown]
# ## 2. Instalar as versões do curso 🔧
# Estas são as versões **testadas juntas** (out/2026). Se o Colab pedir *Restart session*, aceite e siga da seção 3.

# %% tags=["colab-only"]
# %pip install -q scanpy==1.12.4 anndata==0.13.4 mudata==0.4.1 muon==0.1.9 scvi-tools==1.5.1 \
#     snapatac2==2.10.0 celltypist==1.7.1 decoupler==2.2.0 scikit-learn==1.7.2 \
#     scrublet==0.2.3 doubletdetection==4.3.0.post1 scanorama==1.7.4 bbknn==1.6.0 harmonypy==2.0.2 \
#     pychromvar==0.0.4 pyjaspar==4.0.0 macs3==3.0.5 leidenalg==0.12.0 igraph \
#     snakemake==9.27.0 papermill session-info2

# %% [markdown]
# ## 3. Checagem: tudo importa na versão certa?

# %%
import importlib
import pandas as pd

ESPERADO = {
    "scanpy": "1.12.4", "anndata": "0.13.4", "mudata": "0.4.1", "muon": "0.1.9", "scvi": "1.5.1",
    "snapatac2": "2.10.0", "celltypist": "1.7.1", "decoupler": "2.2.0", "sklearn": "1.7.2",
    "scrublet": None, "doubletdetection": None, "scanorama": None, "bbknn": None, "harmonypy": "2.0.2",
    "pychromvar": None, "pyjaspar": None, "MACS3": None, "leidenalg": None, "snakemake": None, "torch": None,
}
linhas = []
for mod, ver in ESPERADO.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        ok = "✅" if (ver is None or str(v).startswith(ver)) else f"⚠️ esperado {ver}"
    except Exception as e:
        v, ok = "-", f"❌ {type(e).__name__}"
    linhas.append((mod, v, ok))
tabela = pd.DataFrame(linhas, columns=["pacote", "versão", "status"]).set_index("pacote")
tabela

# %%
import torch
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2000, 2000, device="cuda"); print("teste GPU ok:", float((x @ x).mean()) is not None)

# %% [markdown]
# ## 4. (Recomendado) Guardar resultados no Google Drive
# Os notebooks salvam em `/content/resultados`. Para não perder nada entre as aulas, aponte para o Drive:

# %%
USAR_DRIVE = False
if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["CURSO_OUT"] = "/content/drive/MyDrive/curso_singlecell/resultados"
    os.makedirs(os.environ["CURSO_OUT"], exist_ok=True)
    print("resultados irão para", os.environ["CURSO_OUT"])

# %% [markdown]
# ## 5. Revisão relâmpago: AnnData e MuData
# ```
# AnnData                         MuData
# ├── X        (células × genes)   ├── mod["rna"]   → AnnData
# ├── obs      (metadados células) ├── mod["atac"]  → AnnData
# ├── var      (metadados genes)   ├── obs / obsm / obsp (compartilhados)
# ├── layers   (ex.: "counts")     └── uns
# ├── obsm     (PCA, UMAP, scVI)
# ├── obsp     (grafos kNN)
# └── uns      (parâmetros, cores)
# ```

# %%
import scanpy as sc, numpy as np, anndata as ad, mudata as md
a = sc.datasets.pbmc68k_reduced()
b = ad.AnnData(np.random.poisson(1, (a.n_obs, 50)).astype(np.float32), obs=a.obs[[]].copy())
b.var_names = [f"chr1:{i*1000}-{i*1000+500}" for i in range(50)]
mdata = md.MuData({"rna": a, "atac": b})
mdata
