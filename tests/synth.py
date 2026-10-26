"""Dados sintéticos para testar os notebooks offline (CI / sem internet).

NÃO são usados em aula: os notebooks baixam dados públicos da 10x Genomics.
Aqui geramos dados com a mesma *estrutura* (nomes de genes reais, tipos de PBMC,
lotes, picos com motivos) para exercitar todas as chamadas de API.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

MARK = {
    "CD4 T": ["CD3E", "CD3D", "CD4", "IL7R", "CCR7", "LEF1", "TCF7", "GATA3"],
    "CD8 T": ["CD3E", "CD3D", "CD8A", "CD8B", "GZMK", "CCL5", "EOMES"],
    "NK": ["GNLY", "NKG7", "KLRD1", "PRF1", "GZMB", "NCAM1", "TBX21", "EOMES"],
    "B": ["MS4A1", "CD79A", "CD79B", "BANK1", "PAX5", "EBF1"],
    "CD14 Mono": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "SPI1", "CEBPA", "CEBPB"],
    "CD16 Mono": ["FCGR3A", "MS4A7", "LST1", "CDKN1C", "SPI1", "CEBPB"],
    "DC": ["FCER1A", "CST3", "CLEC10A", "CD1C", "IRF8", "SPI1"],
}
TFS = ["SPI1", "CEBPA", "CEBPB", "PAX5", "EBF1", "TBX21", "EOMES", "TCF7", "LEF1", "GATA3", "IRF8", "CTCF"]
# TF -> motivo consenso (para inserir nos picos e na fasta sintética)
MOTIFS = {"SPI1": "AAAAAGAGGAAGTG", "CEBPB": "ATTGCGCAAT", "PAX5": "GCGTGACCA",
          "EBF1": "TCCCAAGGGA", "TBX21": "AGGTGTGA", "TCF7": "AGATCAAAG", "CTCF": "CCGCGNGGNGGCAG"}
CT_TF = {"CD4 T": ["TCF7", "LEF1", "GATA3"], "CD8 T": ["TCF7", "EOMES"], "NK": ["TBX21", "EOMES"],
         "B": ["PAX5", "EBF1"], "CD14 Mono": ["SPI1", "CEBPB", "CEBPA"],
         "CD16 Mono": ["SPI1", "CEBPB"], "DC": ["SPI1", "IRF8"]}


def gene_list(n_filler=1200):
    g = sorted({x for v in MARK.values() for x in v} | set(TFS) | {"PPBP", "PF4", "LILRA4", "IL3RA"})
    g += [f"MT-ND{i}" for i in range(1, 6)] + [f"RPL{i}" for i in range(3, 13)]
    g += [f"GENE{i:04d}" for i in range(n_filler)]
    return g


def sim_rna(n=1500, batch_scale=1.0, seed=0, genes=None, types=None):
    rng = np.random.default_rng(seed)
    genes = genes or gene_list()
    types = types or list(MARK)
    G = len(genes); gi = {g: i for i, g in enumerate(genes)}
    base = rng.gamma(0.6, 0.6, G)
    gene_batch = rng.lognormal(0, 0.35 * (batch_scale != 1.0), G)  # efeito de lote por gene
    labels = rng.choice(types, n, p=np.r_[[0.3, 0.15, 0.12, 0.12, 0.2, 0.06, 0.05]][: len(types)] /
                        np.r_[[0.3, 0.15, 0.12, 0.12, 0.2, 0.06, 0.05]][: len(types)].sum())
    mu = np.tile(base, (n, 1))
    prog_rng = np.random.default_rng(12345)          # programas fixos entre lotes
    progs = {t: prog_rng.choice(G, 40, replace=False) for t in MARK}
    for t in types:
        idx = labels == t
        for g in MARK[t]:
            mu[idx, gi[g]] += rng.uniform(4, 9)
        # programas "latentes" por tipo (genes filler)
        mu[np.ix_(idx, progs[t])] *= 3
    lib = rng.lognormal(np.log(1500 * batch_scale), 0.3, n)
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None] * gene_batch[None, :]
    X = rng.negative_binomial(5, 5 / (5 + mu)).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"true_type": labels},
                   index=[f"cell{seed}_{i}-1" for i in range(n)]), var=pd.DataFrame(index=genes))
    a.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(G)]
    return a


def celltypist_models():
    import celltypist, scanpy as sc
    ref = sim_rna(1200, seed=99)
    sc.pp.normalize_total(ref, target_sum=1e4); sc.pp.log1p(ref)
    m = celltypist.train(ref, labels="true_type", max_iter=200, n_jobs=1, check_expression=False)
    return m, m


# --------------------------------------------------------------------- ATAC
CHROMS = {"chr1": 2_000_000, "chr2": 1_500_000}


def write_genome(outdir: Path, genes=None, seed=0):
    """Fasta + GTF sintéticos; cada gene com TSS conhecido."""
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    seqs = {c: rng.choice(list("ACGT"), L) for c, L in CHROMS.items()}
    genes = genes or sorted({x for v in MARK.values() for x in v} | set(TFS))
    rows, tss = [], {}
    for i, g in enumerate(genes):
        c = "chr1" if i % 2 == 0 else "chr2"
        s = 60_000 + (i // 2) * 40_000
        e = s + 5_000
        tss[g] = (c, s)
        attr = f'gene_id "ENSG{i:011d}"; gene_name "{g}"; gene_type "protein_coding"; ' \
               f'transcript_id "ENST{i:011d}"; transcript_name "{g}-201"; transcript_type "protein_coding";'
        rows.append(f"{c}\tsynth\tgene\t{s}\t{e}\t.\t+\t.\t{attr}")
        rows.append(f"{c}\tsynth\ttranscript\t{s}\t{e}\t.\t+\t.\t{attr}")
        rows.append(f"{c}\tsynth\texon\t{s}\t{s+500}\t.\t+\t.\t{attr}")
    # picos: 3 por gene (promotor + 2 enhancers) + picos de fundo
    peaks = []
    for g, (c, s) in tss.items():
        for off in (-200, -15_000, 22_000):
            peaks.append((c, s + off, s + off + 500, g))
    for c, L in CHROMS.items():
        for p in rng.integers(10_000, L - 10_000, 250):
            peaks.append((c, int(p), int(p) + 500, None))
    peaks = sorted(set(peaks), key=lambda x: (x[0], x[1]))
    # inserir motivos de TF nos picos ligados a genes do tipo celular correspondente
    for (c, s, e, g) in peaks:
        if g is None:
            continue
        for t, tfs in CT_TF.items():
            if g in MARK[t]:
                for tf in tfs:
                    if tf in MOTIFS:
                        m = MOTIFS[tf].replace("N", "G")
                        pos = s + rng.integers(50, 400)
                        seqs[c][pos:pos + len(m)] = list(m)
    with open(outdir / "genome.fa", "w") as f:
        for c, sq in seqs.items():
            f.write(f">{c}\n")
            s = "".join(sq)
            for i in range(0, len(s), 80):
                f.write(s[i:i + 80] + "\n")
    with gzip.open(outdir / "genes.gtf.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    return outdir / "genome.fa", outdir / "genes.gtf.gz", peaks, tss


def sim_fragments(path: Path, peaks, n_cells=800, seed=0, types=None, sort_by="barcode"):
    """Fragmentos por célula: picos do tipo celular são enriquecidos + fundo uniforme."""
    rng = np.random.default_rng(seed)
    types = types or list(MARK)
    labels = rng.choice(types, n_cells)
    out = []
    peak_arr = [(c, s, e, g) for (c, s, e, g) in peaks]
    for i, t in enumerate(labels):
        bc = f"AAAC{i:08d}-1"
        open_peaks = [p for p in peak_arr if p[3] is not None and p[3] in MARK[t]]
        shared = [p for p in peak_arr if p[3] is None or p[3] in ("CTCF",)]
        nfrag = int(rng.lognormal(np.log(6000), 0.3))
        for _ in range(nfrag):
            r = rng.random()
            if r < 0.30 and open_peaks:
                c, s, e, _g = open_peaks[rng.integers(len(open_peaks))]
            elif r < 0.55:
                c, s, e, _g = shared[rng.integers(len(shared))]
            else:
                c = "chr1" if rng.random() < 0.57 else "chr2"
                s = int(rng.integers(1000, CHROMS[c] - 2000)); e = s + 500
            st = int(rng.integers(s, e)); ln = int(rng.choice([60, 80, 120, 190, 210, 380]))
            out.append((c, st, st + ln, bc, 1))
    df = pd.DataFrame(out, columns=["chrom", "start", "end", "bc", "n"])
    df = df.sort_values(["bc", "chrom", "start"] if sort_by == "barcode" else ["chrom", "start"])
    df.to_csv(path, sep="\t", header=False, index=False, compression="gzip")
    return pd.Series(labels, index=[f"AAAC{i:08d}-1" for i in range(n_cells)])


def jaspar_like_motifs():
    """Motivos (Bio.motifs) a partir dos consensos, no formato que pychromvar aceita."""
    from Bio import motifs
    from Bio.Seq import Seq
    out = []
    for tf, cons in MOTIFS.items():
        cons = cons.replace("N", "G")
        m = motifs.create([Seq(cons)] * 18 + [Seq("".join(np.random.default_rng(0).choice(list("ACGT"), len(cons))))] * 2)
        m.name = tf; m.matrix_id = f"MA_{tf}"
        out.append(m)
    return out


def snap_motifs():
    import snapatac2 as snap
    lines = []
    for tf, cons in MOTIFS.items():
        cons = cons.replace("N", "G")
        lines.append(f"MOTIF {tf}_synth {tf}\nletter-probability matrix: alength= 4 w= {len(cons)}")
        for ch in cons:
            p = [0.04] * 4; p["ACGT".index(ch)] = 0.88
            lines.append(" ".join(f"{x:.3f}" for x in p))
        lines.append("")
    meme = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n" + "\n".join(lines)
    p = Path("synth_motifs.meme"); p.write_text(meme)
    ms = snap.read_motifs(str(p))
    for m in ms:
        m.name = m.id.split("_")[0]
    return ms


def sim_multiome_h5mu(outdir: Path, n_cells=900, seed=0):
    """MuData pareado (rna + atac) com a mesma estrutura do h5 da 10x (interval, gene_ids...)."""
    import mudata as md
    fa, gtf, peaks, tss = write_genome(outdir, seed=seed)
    rng = np.random.default_rng(seed)
    rna = sim_rna(n_cells, seed=seed)
    labels = rna.obs["true_type"].values
    P = len(peaks)
    pnames = [f"{c}:{s}-{e}" for (c, s, e, g) in peaks]
    X = np.zeros((n_cells, P), dtype=np.float32)
    depth = rng.lognormal(np.log(3000), 0.3, n_cells)
    for i, t in enumerate(labels):
        w = np.full(P, 0.3)
        for j, (c, s, e, g) in enumerate(peaks):
            if g is not None and g in MARK[t]:
                w[j] = 6
            elif g == "CTCF":
                w[j] = 3
        X[i] = rng.poisson(w / w.sum() * depth[i] * 0.4)
    atac = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=rna.obs_names),
                      var=pd.DataFrame(index=pnames))
    atac.var["gene_ids"] = pnames; atac.var["feature_types"] = "Peaks"
    atac.var["interval"] = pnames
    rna.var["feature_types"] = "Gene Expression"
    rna.var["interval"] = [f"{tss[g][0]}:{tss[g][1]}-{tss[g][1]+5000}" if g in tss else "NA"
                           for g in rna.var_names]
    mdata = md.MuData({"rna": rna, "atac": atac})
    return mdata, fa, gtf
