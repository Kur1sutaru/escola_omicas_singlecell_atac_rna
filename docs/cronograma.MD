# Cronograma

## Dia 1 · 26/10 (seg) · 18h–21h · scRNA-seq avançado + scATAC-seq

| Horário | Bloco | Material |
|---|---|---|
| 18:00–18:15 | Abertura, objetivos, como o curso funciona | Slides Aula 1 (1–5) |
| 18:15–18:30 | Colab reprodutível: GPU, versões fixadas, Drive | `00_setup_colab` |
| 18:30–18:50 | Teoria: doublets, efeito de lote, integração, anotação | Slides Aula 1 (6–15) |
| 18:50–19:35 | **Prática scRNA-seq**: doublets → integração (Harmony, BBKNN, Scanorama, scVI) → métricas → CellTypist / decoupler / scANVI | `01_scRNA_...` |
| 19:35–19:45 | Intervalo (o scVI pode treinar durante o intervalo) | |
| 19:45–20:05 | Teoria: fragmentos, TSSe, tiles, espectral × LSI, picos, motivos, chromVAR, tópicos, footprinting | Slides Aula 1 (16–26) |
| 20:05–20:50 | **Prática scATAC-seq**: SnapATAC2 do fragmento aos motivos | `02_scATAC_snapatac2` |
| 20:50–21:00 | Discussão e prévia do Dia 2 | Slides Aula 1 (final) |

## Dia 2 · 27/10 (ter) · 18h–21h · Multiome + GRN + reprodutibilidade

| Horário | Bloco | Material |
|---|---|---|
| 18:00–18:15 | Recapitulação + Multiome: o que muda quando RNA e ATAC vêm da mesma célula | Slides Aula 2 (1–6) |
| 18:15–18:30 | Teoria: MuData, WNN, MultiVI/totalVI, ligações pico–gene, Signac × muon | Slides Aula 2 (7–13) |
| 18:30–19:15 | **Prática Multiome** | `03_multiome_integracao` |
| 19:15–19:25 | Intervalo | |
| 19:25–19:40 | Teoria: GRN, eRegulons, SCENIC+ | Slides Aula 2 (14–20) |
| 19:40–20:25 | **Prática GRN** | `04_redes_regulatorias_grn` |
| 20:25–20:50 | Reprodutibilidade: versões, sementes, papermill, Snakemake, Nextflow, FAIR | `05_reprodutibilidade_pipelines` + Slides Aula 2 (21–27) |
| 20:50–21:00 | Encerramento, próximos passos, notebook bônus | `06_bonus_seus_dados_cellranger` |

## Tempos de execução estimados no Colab

| Notebook | Runtime | Tempo total aproximado | Etapa mais longa |
|---|---|---|---|
| 00 | qualquer | 5 min | instalação |
| 01 | T4 GPU | 15–20 min | DoubletDetection (≈ 3 min) e scVI (≈ 3 min) |
| 02 | CPU | 25–30 min | download do hg38 + MACS3 + chromVAR |
| 03 | T4 GPU | 15 min | MultiVI (≈ 5 min) |
| 04 | CPU | 15–20 min | `add_tf_binding` (varredura de motivos) |
| 05 | CPU | 5 min | pipeline Snakemake |
| 06 | T4 GPU | 10 min | scAR |

> **Dica:** peça aos alunos que abram e rodem a instalação do notebook seguinte **durante a teoria**.
