# Dados

Os notebooks 01–04 baixam automaticamente os dados públicos da 10x Genomics para `/content/data` (no Colab),
com cache: rodar de novo não baixa outra vez.

## `exemplo_GSE293526/` (notebook 06)

Saída do `cellranger count` (v9.0.1, referência GRCm39-2024-A, química Single Cell 3' v3) de epitélio lingual de
camundongos C57BL/6 machos, coletado nos tempos circadianos ZT0 e ZT12.

| Arquivo | Conteúdo |
|---|---|
| `filtered_feature_bc_matrix.h5` | 1.716 células × 33.696 genes |
| `raw_feature_bc_matrix.h5` | todos os barcodes (~1,1 milhão), para estimar RNA ambiente |
| `metrics_summary.csv`, `web_summary.html` | laudo de QC do Cell Ranger |

**Fonte:** GEO [GSE293526](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293526)

**Artigo:** Matsu-ura T, Nasu A, Lee S, Yoshida N, Matsuura K, Yasuda M, Nakamura K, Hong CI, Tsuta K.
Circadian clock–gated cell renewal controls time-dependent changes in taste sensitivity.
*Proc Natl Acad Sci U S A* 122(19):e2421421122 (2025). doi:[10.1073/pnas.2421421122](https://doi.org/10.1073/pnas.2421421122)
