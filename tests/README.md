# Testes offline

`run_notebooks.py` executa cada notebook de `src_notebooks/` com:
- células `colab-only` (instalação) puladas;
- células `load` (downloads) substituídas por *fixtures* com dados sintéticos de `synth.py`;
- a célula `parameters` com valores menores (poucas épocas, menos células).

Os dados sintéticos usam nomes de genes reais de PBMC, um genoma de brinquedo (FASTA + GTF) e motivos de TFs inseridos
nos picos de cada tipo celular. Assim, os testes conferem que a biologia "plantada" é recuperada
(ex.: SPI1/CEBPB em monócitos, PAX5/EBF1 em B, TCF7 em T CD4).

O notebook 06 roda com os dados **reais** de `data/exemplo_GSE293526`.
