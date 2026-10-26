import pandas as pd, re
linhas = []
for cl, mk in zip(snakemake.input.cl, snakemake.input.mk):
    res = re.search(r"res([0-9.]+)\.csv", cl).group(1)
    c = pd.read_csv(cl, index_col=0); m = pd.read_csv(mk)
    top = m.groupby("group")["names"].first().to_dict()
    for k, n in c["leiden"].value_counts().sort_index().items():
        linhas.append({"resolucao": res, "cluster": k, "n_celulas": n, "marcador_top": top.get(str(k), top.get(k))})
pd.DataFrame(linhas).to_csv(snakemake.output[0], index=False)
