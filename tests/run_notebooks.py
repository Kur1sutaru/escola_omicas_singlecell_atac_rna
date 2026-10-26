"""Executa os notebooks offline com dados sintéticos.

Uso:  python tests/run_notebooks.py [nome_parcial_do_notebook ...]

- células com tag `colab-only` (instalação) são puladas;
- células com tag `load` (download) são substituídas, na ordem, pelos fixtures abaixo;
- a célula `parameters` recebe sobrescritas para rodar rápido.
"""
import os, sys, time
from pathlib import Path

import nbformat
import jupytext
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tests" / "_work"; WORK.mkdir(parents=True, exist_ok=True)

PRE = f"""
import sys; sys.path.insert(0, r"{ROOT / 'tests'}")
import synth
"""

FIXTURES = {
    "00_": {"params": "", "load": []},
    "01_": {
        "params": "N_CELLS_10K=1500; N_HVG=800; MAX_EPOCHS_SCVI=15; MAX_EPOCHS_SCANVI=3",
        "load": [
            PRE + """
import anndata as ad, scanpy as sc
genes = synth.gene_list()
a3k = synth.sim_rna(900, batch_scale=0.6, seed=1, genes=genes)
a10k = synth.sim_rna(1500, batch_scale=2.5, seed=2, genes=genes)
adata = ad.concat({"pbmc3k_v1": a3k, "pbmc10k_v3": a10k}, label="batch", join="inner", index_unique="-")
""",
            PRE + """
import celltypist
modelo_fino, modelo_grosso = synth.celltypist_models()
""",
            """
net_panglao = net_pbmc.copy()
""",
        ],
    },
    "02_": {
        "params": "N_FEATURES=20000; MIN_COUNTS=500; MIN_TSSE=0.0; N_TOPICOS=5; MACS_QVALUE=0.5",
        "load": [
            PRE + """
import snapatac2 as snap
from pathlib import Path
fa, gtf, peaks, tss = synth.write_genome(DATA / "genoma")
GENOMA = snap.genome.Genome(fasta=fa, annotation=gtf, chrom_sizes=synth.CHROMS)
FRAGMENTOS = DATA / "frags.tsv.gz"
verdade = synth.sim_fragments(FRAGMENTOS, peaks, n_cells=700)
""",
            PRE + """
MOTIVOS_SNAP = synth.snap_motifs()
""",
            PRE + """
MOTIVOS_JASPAR = synth.jaspar_like_motifs()
""",
        ],
    },
    "03_": {
        "params": "MAX_EPOCHS_MULTIVI=5; MIN_CELLS_PEAK=0.0; N_HVG=800; RODAR_TOTALVI=False",
        "load": [
            PRE + """
mdata, FASTA_PATH, GTF_PATH = synth.sim_multiome_h5mu(DATA / "multiome")
""",
            PRE + """
import celltypist
modelo_grosso, _ = synth.celltypist_models()
""",
        ],
    },
    "04_": {
        "params": "N_METACELULAS=40; JANELA=60_000; MIN_CELLS_PEAK=0.0; MIN_ALVOS=3",
        "load": [
            PRE + """
import muon as mu, scanpy as sc, snapatac2 as snap
mdata, fa, gtf = synth.sim_multiome_h5mu(DATA / "multiome")
GENOMA = snap.genome.Genome(fasta=fa, annotation=gtf, chrom_sizes=synth.CHROMS)
for m in ("rna", "atac"):
    mdata.mod[m].layers["counts"] = mdata.mod[m].X.copy()
mdata.obs["cell_type"] = mdata.mod["rna"].obs["true_type"].values
mdata.mod["rna"].obs["cell_type"] = mdata.obs["cell_type"].values
""",
            PRE + """
import pandas as pd
rows = [(tf, g, 1) for t, tfs in synth.CT_TF.items() for tf in tfs for g in synth.MARK[t] if g != tf]
collectri = pd.DataFrame(rows, columns=["source", "target", "weight"]).drop_duplicates(["source", "target"])
""",
            PRE + """
MOTIVOS_SNAP = synth.snap_motifs()
MOTIVOS_JASPAR = synth.jaspar_like_motifs()
""",
        ],
    },
    "06_": {
        "params": "MAX_EPOCHS_SCAR = 15",
        "load": [f"""
from pathlib import Path
OUTS = Path(r"{ROOT / 'data' / 'exemplo_GSE293526'}")
"""],
    },
    "05_": {"params": PRE + """
_a = synth.sim_rna(600, seed=5); _p = DATA / "synth_raw.h5ad"; _a.write_h5ad(_p); ENTRADA_LOCAL = str(_p)
""", "load": []},
}


def prepare(nb_path: Path, fx: dict) -> nbformat.NotebookNode:
    nb = jupytext.read(nb_path)
    loads = iter(fx["load"])
    cells = []
    for c in nb.cells:
        tags = c.get("metadata", {}).get("tags", [])
        if c.cell_type != "code":
            continue
        if "colab-only" in tags:
            continue
        if "load" in tags:
            c.source = next(loads)
        if "parameters" in tags and fx["params"]:
            c.source += "\n" + fx["params"]
        cells.append(c)
    nb.cells = cells
    return nb


def main(filters):
    os.environ["CURSO_DATA"] = str(WORK / "data")
    os.environ["CURSO_OUT"] = str(WORK / "out")
    os.environ["MPLBACKEND"] = "Agg"
    ok = True
    for src in sorted((ROOT / "src_notebooks").glob("*.py")):
        key = src.name[:3]
        if key not in FIXTURES or (filters and not any(f in src.name for f in filters)):
            continue
        nb = prepare(src, FIXTURES[key])
        t = time.time()
        client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(WORK)}})
        try:
            client.execute()
            print(f"OK   {src.name}  ({time.time() - t:.0f}s)")
        except Exception as e:
            ok = False
            import re; print(f"FAIL {src.name}\n{re.sub(chr(27)+r'\[[0-9;]*m', '', str(e))[-3000:]}")
        nbformat.write(nb, WORK / (src.stem + ".executed.ipynb"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
