---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #4C72B0;
  }
  table {
    font-size: 19px;
  }
---

# Diagnóstico de Câncer de Mama
## EDA, Seleção de Features, Treinamento e Avaliação do Modelo

Equipe 5: Rodrigo Torres, Nicolas Pinheiro, Daniel, Samuel Lima e Roan

Probabilidade e Estatística

Dataset: UCI Machine Learning Repository (ID 17) — Breast Cancer Wisconsin (Diagnostic)

---

# Parte 1 — Análise Exploratória (EDA)

## Estrutura do Dataset

| Propriedade | Valor |
|---|---|
| Observações | 569 |
| Variáveis preditoras | 30 (todas contínuas/reais) |
| Variável-alvo | `Diagnosis`: **B** (Benigno) ou **M** (Maligno) |
| Valores ausentes | 0 |
| Tipo de problema | Classificação binária |

Nenhum valor nulo → não foi necessária nenhuma estratégia de imputação (remoção de linha, remoção de coluna, preenchimento por média/mediana).

---

## As 10 Características-Base

| Característica | O que mede |
|---|---|
| `radius` | Distância média do centro aos pontos do perímetro |
| `texture` | Desvio padrão da escala de cinza (textura da célula) |
| `perimeter` / `area` | Tamanho do núcleo |
| `smoothness` | Variação local no contorno |
| `compactness` | perímetro² / área − 1.0 |
| `concavity` | Severidade das reentrâncias do contorno |
| `concave points` | Número de reentrâncias do contorno |
| `symmetry` | Simetria do núcleo |
| `fractal dimension` | Irregularidade do contorno |

---

## Por que 30 variáveis, e não 10?

Cada característica-base foi calculada de **3 formas** por imagem (várias células por imagem):

| Sufixo | Significado |
|---|---|
| `mean` | Média dos valores de todos os núcleos da imagem |
| `error` | Erro padrão (variabilidade entre núcleos) |
| `worst` | Média dos 3 piores (maiores) valores observados |

**10 características × 3 formas = 30 features.** Ex.: `mean radius`, `radius error`, `worst radius` são a mesma medida, vista sob 3 ângulos estatísticos diferentes.

---

## Gráfico 1 — Distribuição das Classes

![bg right:60% width:600px](graficos/01_distribuicao_classes.png)

- **Benigno**: 357 casos (62,7%)
- **Maligno**: 212 casos (37,3%)

---

## Gráfico 2 — Histogramas das Features

![bg right:55% width:650px](graficos/02_histogramas_features.png)

- Distribuição de cada uma das 30 features, separada por classe (azul = benigno, vermelho = maligno)
- Onde as curvas se separam bem, a feature tem poder discriminativo
- Features de tamanho (`area`, `perimeter`, `radius`) mostram pouca sobreposição entre classes — já são candidatas fortes

---

## Gráfico 3 — Boxplots por Classe

![bg right:55% width:550px](graficos/03_boxplots_top10_features.png)

Top 10 features com **maior diferença de médias** entre Benigno e Maligno:

`worst area`, `mean area`, `worst perimeter`, `area error`, `mean perimeter`, `worst radius`, `worst texture`, `mean radius`, `mean texture`, `perimeter error`

Lista dominada por features de **tamanho**. Caixas bem separadas = boa capacidade de distinguir classes sozinha.

---

## Gráfico 4 — Heatmap de Correlação

![bg right:55% width:650px](graficos/04_heatmap_correlacao.png)

- Correlação de Pearson entre as 30 features
- **21 pares** com correlação absoluta ≥ 0.90
- Exemplos: `mean radius` ↔ `mean perimeter` (r = 0.998), `worst radius` ↔ `worst perimeter` (r = 0.994), `mean radius` ↔ `mean area` (r = 0.987)

Revela **multicolinearidade**: as features de tamanho medem essencialmente a mesma coisa.

---

## Gráfico 5 — Pairplot das Top 5 Features

![bg right:55% width:650px](graficos/05_pairplot_top5_features.png)

As 5 features mais correlacionadas com o diagnóstico:

| Feature | \|correlação\| |
|---|---|
| `worst concave points` | 0.79 |
| `worst perimeter` | 0.78 |
| `mean concave points` | 0.78 |
| `worst radius` | 0.78 |
| `mean perimeter` | 0.74 |

Nos scatterplots, benignos e malignos formam agrupamentos visualmente separados.

---

## Conclusões da EDA

- Dataset limpo (sem nulos) e moderadamente balanceado
- Features de **tamanho** e **forma irregular** (`concave points`, `concavity`) são as mais discriminativas
- Escalas muito diferentes entre features → padronização será necessária
- Alta multicolinearidade (21 pares r ≥ 0.90) → problema relevante para o próximo passo, já que o algoritmo exigido (Naive Bayes) assume independência entre variáveis

---

# Parte 2 — Seleção de Features e Treinamento

---

## Restrição do Trabalho

> O enunciado do PDF exige especificamente o algoritmo **Naive Bayes**.

- Não era permitido trocar de algoritmo para contornar o problema de multicolinearidade
- Todo o espaço de melhoria ficou concentrado em: **quais features usar** e **como calibrar as probabilidades de saída**
- Algoritmo usado em todos os experimentos: `GaussianNB` (assume que cada feature, dado a classe, segue uma distribuição normal)

---

## Por Que a Seleção de Features Importa Aqui

O Naive Bayes assume que as variáveis são **condicionalmente independentes** dado a classe — ele multiplica as probabilidades de cada feature como se elas não tivessem relação entre si.

A EDA mostrou o contrário: **21 pares de features com correlação ≥ 0.90**.

**Consequência prática:** o modelo pode "contar a mesma evidência" várias vezes (ex: `radius`, `perimeter` e `area` juntos apontando para o mesmo sinal de tamanho), gerando probabilidades exageradamente extremas — inclusive em previsões erradas.

---

## Pré-processamento Aplicado

Pipeline único usado em todas as 4 abordagens:

```
StandardScaler  →  GaussianNB
```

1. **Validação**: confirma ausência de valores nulos
2. **Split treino/teste**: 80% / 20%, com `stratify` (mantém a proporção 63%/37% em ambos os conjuntos)
3. **`random_state=42`** fixo, para resultados reprodutíveis
4. **`StandardScaler`**: padroniza cada feature para média 0 e desvio padrão 1 — resolve o problema de escalas diferentes visto na EDA

---

## Como as Features Foram Selecionadas

**Método 1 — SelectKBest (ANOVA F-classif):**
Calcula, para cada feature, o quanto a média dela difere entre as classes Benigno/Maligno, relativo à variância interna de cada classe. Quanto maior o F-score, mais discriminativa.

**Método 2 — Filtro de baixa correlação:**
Parte do ranking ANOVA, mas ao adicionar uma feature verifica se ela tem correlação ≥ 0.90 com alguma já escolhida — se tiver, é descartada e passa para a próxima do ranking.

Ambos os métodos são calculados **somente no conjunto de treino**, evitando vazamento de dados (data leakage) do conjunto de teste.

---

## 4 Abordagens Testadas

| # | Abordagem | Nº features | Critério |
|---|---|---|---|
| 1 | Baseline | 30 | Nenhuma seleção |
| 2 | **Seleção estatística** (padrão) | 10 | Top-10 por score ANOVA |
| 3 | Baixa correlação | 12 | ANOVA + remove pares r ≥ 0.90 |
| 4 | Calibrado + baixa correlação | 12 | Igual a (3) + calibração sigmoid |

---

## Abordagem 2 (Padrão) — As 10 Features Escolhidas

`worst concave points`, `worst perimeter`, `worst radius`, `mean concave points`, `mean perimeter`, `worst area`, `mean radius`, `mean area`, `mean concavity`, `worst concavity`

**Isso bate com a EDA?** Sim — as 5 features mais correlacionadas com o target identificadas no pairplot (slide anterior) estão **todas presentes** nessa lista de 10.

Motivo matemático: o score ANOVA F (treino) e a correlação ponto-bisserial (EDA) são estatisticamente equivalentes para um alvo binário — a EDA já havia "previsto" essas features.

---

## Abordagem 3/4 — As 12 Features de Baixa Correlação

`worst concave points`, `worst perimeter`, `mean concavity`, `worst concavity`, `mean compactness`, `worst compactness`, `radius error`, `worst texture`, `worst smoothness`, `worst symmetry`, `concave points error`, `mean smoothness`

**Por que essa lista parece tão diferente do boxplot da EDA?**
O boxplot é dominado por features de tamanho (`area`, `perimeter`, `radius`) — justamente as mais correlacionadas entre si. O filtro de baixa correlação descarta a maioria delas e as troca por `concavity`, `compactness`, `symmetry`, `smoothness`. **Não é inconsistência — é o resultado esperado do problema identificado na EDA.**

---

## Calibração de Probabilidades (Abordagem 4)

- `GaussianNB` puro tende a gerar probabilidades muito extremas (perto de 0 ou 1), mesmo quando erra
- A **calibração sigmoid** (`CalibratedClassifierCV`) reajusta essas probabilidades para refletir melhor a confiança real do modelo
- Objetivo: se o modelo diz "95% de chance de maligno", isso deveria realmente acontecer em ~95% dos casos parecidos
- Medido pelo **Brier score** (quanto menor, melhor calibrado)

---

# Parte 3 — Avaliação do Modelo

---

## Métricas Utilizadas — O Que Cada Uma Mede

| Métrica | O que mede |
|---|---|
| **Acurácia** | % total de acertos |
| **Precisão** | Dos casos previstos como X, quantos realmente eram X |
| **Recall** | Dos casos reais de X, quantos o modelo identificou |
| **F1-score** | Média harmônica entre precisão e recall |
| **Brier score** | Qualidade da probabilidade prevista (menor = melhor) |
| **Log loss** | Penaliza fortemente probabilidades erradas e "confiantes" |

**Macro** = calculado por classe e depois tirada a média simples (não pondera pelo tamanho da classe) — importante aqui porque as classes são desbalanceadas.

---

## Resultados Completos dos 4 Experimentos

| Abordagem | Acurácia | F1 macro | Recall Maligno | Brier Maligno |
|---|---|---|---|---|
| Baseline (30) | 0.921 | 0.914 | 0.857 | 0.0757 |
| **Seleção estatística (10)** | **0.939** | **0.933** | **0.881** | **0.0542** |
| Baixa correlação (12) | 0.895 | 0.886 | 0.833 | 0.0952 |
| Calibrado + baixa correlação (12) | 0.886 | 0.876 | 0.810 | 0.0904 |

A abordagem "Seleção estatística" venceu em **todas** as métricas prioritárias simultaneamente.

---

## Critério de Escolha do Modelo Final

Não foi decidido só por acurácia. Ordem de prioridade:

1. **F1 macro** — equilíbrio entre precisão e recall nas duas classes
2. **Recall da classe maligna** — o erro mais grave é classificar câncer como benigno (falso negativo)
3. **Brier score** — qualidade/calibração das probabilidades
4. **Confiança média nos erros** — o quão "convicto" o modelo fica quando erra (idealmente baixo)

---

## Achado Científico Interessante

Reduzir a multicolinearidade (abordagem "baixa correlação") **piorou** o desempenho: **89,5%** vs **93,9%** de acurácia — mesmo sendo teoricamente mais correto para as premissas do Naive Bayes.

**Hipótese:** as features de tamanho removidas por serem redundantes ainda carregavam sinal preditivo relevante. O ganho teórico de "mais independência entre variáveis" não compensou a perda de informação.

➡️ "Mais correto na teoria" nem sempre significa "melhor na prática" — reforça a importância de validar empiricamente.

---

## Relatório de Classificação — Modelo Escolhido

114 casos de teste (20% dos dados):

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---|---|---|---|
| Tumor Benigno | 0.933 | 0.972 | 0.952 | 72 |
| Tumor Maligno | 0.949 | 0.881 | 0.914 | 42 |
| **Média Macro** | 0.941 | 0.927 | 0.933 | 114 |

O modelo é melhor em identificar Benignos (recall 97,2%) do que Malignos (recall 88,1%) — ponto de atenção clínica.

---

## Matriz de Confusão — Modelo Escolhido

|  | Previsto Benigno | Previsto Maligno |
|---|---|---|
| **Real Benigno** | 70 | 2 |
| **Real Maligno** | 5 | 37 |

- **5 falsos negativos**: tumores malignos classificados como benignos — o erro clinicamente mais grave (atrasaria tratamento)
- **2 falsos positivos**: tumores benignos classificados como malignos — gera alarme desnecessário, porém menos grave
- Log loss: 0.240 · Brier maligno: 0.054 · Confiança média nos erros: 91,3%

---

## Validação Cruzada (5-fold Estratificada)

Para confirmar que o resultado não depende de um único split treino/teste:

| Métrica | Valor médio (CV) | Valor no teste único |
|---|---|---|
| Acurácia | 94,0% | 93,9% |
| F1 macro | 93,6% | 93,3% |
| Recall macro | 93,6% | 92,7% |

Números muito próximos entre CV e teste único → reforça que o resultado é estável, não "sorte" de um split específico.

---

## Limiar de Decisão (Threshold) Ajustável

- Por padrão, o modelo classifica como Maligno se `P(Maligno) ≥ 0.5`
- A API (`/predict`) e o Frontend permitem **ajustar esse limiar**
- Baixar o limiar (ex: 0.3) → modelo fica mais "cauteloso", aumenta recall de maligno mas gera mais falsos positivos
- Em contexto clínico real, esse tipo de ajuste costuma favorecer recall (evitar falsos negativos), mesmo às custas de mais falsos alarmes