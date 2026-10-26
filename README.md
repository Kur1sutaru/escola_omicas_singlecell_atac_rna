# Análise Avançada de Dados de Single-cell RNA-seq, ATAC-seq e Multiome (10x Genomics) no Google Colab

**Ministrante:** Dra. Cristal Villalba · [github.com/kur1sutaru](https://github.com/kur1sutaru)
**Datas:** 26 e 27 de outubro de 2026, 18h–21h (6 h no total)
**Formato:** aulas expositivas + práticas guiadas em Google Colab, com dados públicos da 10x Genomics

Curso de aprofundamento para quem já cursou o módulo introdutório de single-cell. O foco é **ferramental**:
pacotes e estratégias para processamento avançado de scRNA-seq, scATAC-seq e Multiome pareado (10x Genomics),
incluindo integração multiômica, inferência de redes regulatórias e boas práticas de reprodutibilidade.

---

## Notebooks

| # | Notebook | Aula | Colab | GPU |
|---|---|---|---|---|
| 00 | Preparando o Colab (versões, GPU, Drive) | Dia 1 | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/00_setup_colab.ipynb) | – |
| 01 | scRNA-seq: doublets, integração e anotação | Dia 1 | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/01_scRNA_doublets_integracao_anotacao.ipynb) | recomendada |
| 02 | scATAC-seq com SnapATAC2: picos, motivos, chromVAR, tópicos | Dia 1 | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/02_scATAC_snapatac2.ipynb) | não |
| 03 | Multiome: MuData, WNN, MultiVI, ligações pico–gene | Dia 2 | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/03_multiome_integracao.ipynb) | recomendada |
| 04 | Redes regulatórias: eRegulons TF → enhancer → gene (+ roteiro SCENIC+) | Dia 2 | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/04_redes_regulatorias_grn.ipynb) | não |
| 05 | Reprodutibilidade: versões, sementes, papermill, Snakemake, Nextflow, FAIR | Dia 2 | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/05_reprodutibilidade_pipelines.ipynb) | não |
| 06 | **Bônus:** seus dados, do output do Cell Ranger à análise (RNA ambiente com scAR) | extra | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kur1sutaru/curso-singlecell-multiome-avancado/blob/main/notebooks/06_bonus_seus_dados_cellranger.ipynb) | recomendada |

Slides: [`slides/`](slides/) · Cronograma detalhado: [`docs/cronograma.md`](docs/cronograma.md) · Referências: [`docs/referencias.md`](docs/referencias.md)

## Conteúdo por tema

**scRNA-seq: integração e anotação avançada**: Scrublet, DoubletDetection e scDblFinder (R, opcional);
Harmony, BBKNN, Scanorama e scVI/scANVI com métricas de mistura × conservação; CellTypist (+ *majority voting*),
*label transfer* (scANVI e `sc.tl.ingest`) e *scoring* de assinaturas com decoupler.

**scATAC-seq**: fragmentos, TSS enrichment, matriz de *tiles*, SnapATAC2 (espectral) × TF-IDF/LSI (muon) × episcanpy;
picos com MACS3 por tipo celular; *gene activity*; enriquecimento de motivos e chromVAR; tópicos (LDA, modelo do cisTopic);
coverage por grupo e roteiro de footprinting com TOBIAS.

**Multiome**: MuData/muon, WNN, MultiVI (e onde entra o totalVI), metacélulas e ligações pico–gene com nulo empírico,
equivalências com Signac/Seurat.

**GRN**: atividade de TFs (decoupler + CollecTRI), rede região→gene (SnapATAC2: correlação + elastic net),
TF→região por motivos, eRegulons, validação externa contra o CollecTRI, AUCell e priorização de alvos;
roteiro do SCENIC+ em ambiente Python 3.11 isolado.

**Reprodutibilidade**: versões fixadas, sementes, papermill, checksums, Snakemake (pipeline completo em
[`pipelines/snakemake`](pipelines/snakemake)), visão geral de Nextflow/nf-core e publicação FAIR.

## Dados

| Notebook | Dataset | Fonte |
|---|---|---|
| 01 | PBMC 3k (3' v1) + PBMC 10k (3' v3) | 10x Genomics |
| 02 | PBMC 5k scATAC-seq | 10x Genomics (via `snapatac2.datasets`) |
| 03, 04 | PBMC granulócitos removidos 3k, Multiome ARC v2.0 | 10x Genomics |
| 06 | Epitélio lingual de camundongo, Cell Ranger 9.0.1 | GEO [GSE293526](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293526) · [`data/exemplo_GSE293526`](data/exemplo_GSE293526) |

Os downloads são feitos pelos próprios notebooks (com cache). Nada precisa ser baixado manualmente.

## Ambiente

As versões abaixo foram **testadas juntas** (outubro de 2026) e estão fixadas em cada notebook e em
[`environment/requirements-colab.txt`](environment/requirements-colab.txt):

`scanpy 1.12.4 · anndata 0.13.4 · mudata 0.4.1 · muon 0.1.9 · scvi-tools 1.5.1 · snapatac2 2.10.0 · celltypist 1.7.1 ·
decoupler 2.2.0 · scikit-learn 1.7.2 · harmonypy 2.0.2 · pychromvar 0.0.4 · macs3 3.0.5 · snakemake 9.27.0`

### Incompatibilidades conhecidas (e como os notebooks lidam com elas)

| Problema | Solução adotada |
|---|---|
| `sc.external.pp.harmony_integrate` + harmonypy 2.0: erro de dimensão | chamada direta ao `harmonypy.run_harmony` com checagem de forma |
| CellTypist 1.7 + scikit-learn ≥ 1.8: `multi_class` removido | scikit-learn fixado em 1.7.2 |
| episcanpy 0.4 não importa com numpy 2 / anndata ≥ 0.11 | demonstrado como lição; *shims* só para demonstração |
| `pychromvar.compute_deviations` + anndata ≥ 0.11 (e uso de memória denso) | reimplementação esparsa do cálculo de desvios, validada contra o original |
| `snapatac2.pl.*(interactive=False)` exige Chrome (kaleido) | gráficos interativos (plotly), padrão do Colab |

## Testes

Todos os notebooks rodam **offline** com dados sintéticos de mesma estrutura (nomes de genes reais, picos com motivos inseridos)
e o notebook bônus roda com os dados reais do GEO:

```bash
pip install -r environment/requirements-colab.txt jupytext nbclient ipykernel papermill biopython
python tests/run_notebooks.py          # todos
python tests/run_notebooks.py 03_      # só um
```

As fontes dos notebooks ficam em [`src_notebooks/`](src_notebooks/) (formato *py:percent*, fácil de revisar no git);
os `.ipynb` são gerados com `bash scripts/build_notebooks.sh`.

## Estrutura

```
├── notebooks/            # .ipynb para abrir no Colab
├── src_notebooks/        # fontes jupytext (py:percent)
├── slides/               # Aula1.pptx, Aula2.pptx
├── pipelines/
│   ├── snakemake/        # pipeline scRNA-seq mínimo (notebook 05)
│   └── nextflow/         # esboço DSL2 equivalente
├── environment/          # requirements e environment.yml
├── data/                 # exemplo GSE293526 (notebook 06) + instruções
├── docs/                 # cronograma, referências, guia da instrutora
├── tests/                # dados sintéticos + executor offline
└── scripts/              # utilitários (build dos notebooks)
```

## Como citar

Veja [`CITATION.cff`](CITATION.cff). Código sob licença MIT; textos, slides e figuras sob CC BY 4.0 ([`LICENSE`](LICENSE)).
