# Guia da instrutora

Notas práticas para conduzir as aulas, incluindo o que pode dar errado e como contornar ao vivo.

## Antes do curso (semana de 19/10)
- [ ] Criar o repositório público `kur1sutaru/curso-singlecell-multiome-avancado` e conferir os badges "Abrir no Colab".
- [ ] Rodar **todos** os notebooks no Colab com GPU T4, do início ao fim, e anotar os tempos reais.
- [ ] Se algum download da 10x mudar de URL, atualizar a célula 📥 correspondente.
- [ ] Enviar aos alunos, 2 dias antes: link do repo + pedido para rodar o `00_setup_colab` e conferir a tabela ✅.
- [ ] Ter um plano B: rodar os notebooks 01–04 com antecedência, salvar os `.ipynb` **executados** e deixá-los no Drive
      para quem ficar sem GPU ou sem cota.

## Pontos de atenção por notebook

### 01 · scRNA-seq
- O DoubletDetection imprime um aviso "Leiden clustering is experimental": é esperado.
- **A concordância entre métodos de doublets costuma ser baixa (Jaccard < 0,3).** Use isso como gancho de discussão.
- Se a cota de GPU acabar, `MAX_EPOCHS_SCVI = 50` em CPU ainda dá um resultado razoável (~6 min).
- O Harmony é chamado direto pelo `harmonypy` por causa da incompatibilidade com o wrapper do Scanpy (ver README).
  Bom momento para a mensagem de manutenção de software.

### 02 · scATAC-seq
- O primeiro acesso a `snap.genome.hg38.fasta` baixa ~1 GB (FASTA do hg38). Peça para rodar a seção 9 cedo.
- Os gráficos do SnapATAC2 são **plotly interativos**. `interactive=False` exige Chrome (kaleido) e falha no Colab.
- O `episcanpy` falha de propósito: é a lição sobre ferramentas sem manutenção.
- O chromVAR é uma reimplementação esparsa (validada contra o pychromvar: diferença máx. ~3e-6). O original densifica a
  matriz e estoura a RAM do Colab com dados reais.

### 03 · Multiome
- `mu.read_10x_h5(..., extended=False)` evita procurar fragmentos/anotações que não baixamos.
- O MultiVI pode ser treinado durante a explicação do WNN.
- A métrica de pureza usa rótulos do **RNA**: deixe claro que isso favorece o RNA.

### 04 · GRN
- Carrega o checkpoint do notebook 03 (`aula2_multiome.h5mu`). Se a sessão caiu, a célula refaz um pré-processamento mínimo.
- `add_tf_binding` varre só os motivos de TFs presentes no RNA, para economizar tempo.
- Explique por que o SCENIC+ não roda no Colab gratuito e mostre a seção 8 (ambiente Python 3.11 com `uv`).

### 05 · Reprodutibilidade
- O pipeline Snakemake é escrito pelo próprio notebook (cópia idêntica em `pipelines/snakemake/`).
- Exercício ao vivo: mudar `resolucoes` no `config.yaml` e mostrar que só a nova resolução roda.

### 06 · Bônus
- O diagnóstico do scAR é o ponto central. No epitélio lingual, a "sopa" é dominada por marcadores do próprio epitélio
  (Krt13, Krt14, Krt4), e a descontaminação pode remover sinal real. A regra de decisão automática (`LIMIAR_MANTIDO`)
  escolhe as contagens cruas quando isso acontece.

## Perguntas frequentes dos alunos
- **"Posso usar dados integrados para DE?"** Não: use contagens cruas com lote como covariável (ou pseudobulk + DESeq2/edgeR).
- **"Harmony ou scVI?"** Depende do número de lotes, da composição e do tempo disponível. Mostre a tabela de métricas.
- **"Preciso de picos ou posso usar tiles?"** Tiles para redução de dimensão/clusterização; picos para motivos, ligações e GRN.
- **"Gene activity substitui RNA?"** Não. É um proxy ruidoso; serve para anotação grosseira.
