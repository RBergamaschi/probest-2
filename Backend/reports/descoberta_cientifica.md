# Pipeline de Descoberta Cientifica

## Restricao do trabalho

O PDF exige o uso do algoritmo Naive Bayes. Por isso, todos os experimentos abaixo mantem Naive Bayes como classificador principal. As melhorias foram feitas em selecao de features, reducao de multicolinearidade e calibracao de probabilidades.

## Hipotese inicial

Comecamos com `GaussianNB` usando as 30 features do dataset WDBC. Essa abordagem aproveita toda a informacao da base e atende diretamente ao enunciado.

## Problema observado

O Naive Bayes assume que as variaveis sao independentes. No WDBC, varias features sao fortemente correlacionadas, especialmente raio, perimetro e area em diferentes agregacoes. Isso pode fazer o modelo contar evidencias parecidas mais de uma vez e gerar probabilidades muito extremas, inclusive em erros.

## Experimentos executados

### Baseline - 30 features

Primeiro experimento: GaussianNB com todas as 30 features do WDBC.

- Features usadas: 30
- Acuracia: 0.9211
- F1 macro: 0.9138
- Recall maligno: 0.8571
- Brier maligno: 0.0757
- Confianca media nos erros: 0.9782

### Selecao estatistica - 10 features

GaussianNB usando as 10 features com maior score ANOVA no conjunto de treino.

- Features usadas: 10
- Acuracia: 0.9386
- F1 macro: 0.9330
- Recall maligno: 0.8810
- Brier maligno: 0.0542
- Confianca media nos erros: 0.9128

### Baixa correlacao - 12 features

GaussianNB com features ranqueadas por ANOVA, mas evitando pares com correlacao absoluta acima de 0.90.

- Features usadas: 12
- Acuracia: 0.8947
- F1 macro: 0.8857
- Recall maligno: 0.8333
- Brier maligno: 0.0952
- Confianca media nos erros: 0.9376

### Calibrado + baixa correlacao

GaussianNB com as features de menor redundancia e calibracao sigmoid para probabilidades menos extremas.

- Features usadas: 12
- Acuracia: 0.8860
- F1 macro: 0.8755
- Recall maligno: 0.8095
- Brier maligno: 0.0904
- Confianca media nos erros: 0.8510

## Criterio de escolha

A escolha nao olhou apenas para acuracia. Priorizamos F1 macro, recall da classe maligna e probabilidades menos exageradas, medidas por Brier score e confianca media nos erros.

## Abordagem recomendada

A abordagem recomendada foi `Selecao estatistica - 10 features` (`selectkbest_10_features`).

Motivo:

- Mantem Naive Bayes, portanto respeita o PDF.
- Usa 10 features.
- F1 macro: 0.9330.
- Recall maligno: 0.8810.
- Brier maligno: 0.0542.

## Conclusao

O modelo final nao representa uma verdade clinica. Ele e uma demonstracao academica de classificacao supervisionada com Naive Bayes. A calibracao e a selecao de features ajudam a tornar a probabilidade exibida mais interpretavel, mas falsos positivos e falsos negativos ainda podem ocorrer.
