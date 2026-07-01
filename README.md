# Trabalho Final N2 - Probabilidade e Estatistica

Projeto academico para classificacao de tumores usando o Breast Cancer Wisconsin Diagnostic Dataset (WDBC), Naive Bayes, backend Flask e frontend React.

## Estrutura

- `Backend/`: analise exploratoria, treinamento, modelo salvo, API Flask e testes.
- `Frontend/`: interface React/Vite integrada com a API Flask.
- `probest-2/`: esboco original preservado como referencia.

## Backend

A base e obtida com `ucimlrepo` pelo ID 17:

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=17)
X = dataset.data.features
y = dataset.data.targets
```

Classes:

- `B`: Tumor Benigno
- `M`: Tumor Maligno

Comandos:

```powershell
cd Backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.eda
python -m src.train_model
python -m pytest
python -m app.api
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `GET /examples`
- `GET /feature-stats`
- `GET /model-summary`
- `POST /predict`

Exemplo de `POST /predict`:

```json
{
  "features": {
    "radius1": 17.99,
    "texture1": 10.38
  }
}
```

O objeto `features` deve conter todas as 30 variaveis retornadas por `/model-info`.

## Frontend

O frontend usa Vite/React/TanStack Router diretamente, sem preset externo de scaffold.

Funcionalidades atuais:

- modo simples com exemplos reais da base;
- modo avancado com edicao das 30 features;
- controle de limiar para classificar maligno;
- grafico de probabilidades benigno/maligno;
- alerta visual para predicoes de baixa confianca;
- comparacao das features da amostra contra medias benignas e malignas;
- explicacao textual automatica da predicao;
- painel de desempenho com metricas, baseline e matriz de confusao;
- distribuicao das classes da base.

Comandos:

```powershell
cd Frontend
npm install
npm run dev
```

URL local:

```text
http://127.0.0.1:5173
```

Por padrao, o frontend chama:

```text
http://localhost:5000
```

Para alterar a URL da API, use `VITE_API_BASE_URL`.

## Resultado Do Modelo

As metricas ficam em `Backend/reports/metrics.json`.

## Features E Pre-Processamento

O modelo usa as 30 variaveis preditoras do WDBC, que representam medidas morfologicas do tumor. Essa escolha mantem toda a informacao disponibilizada pela base e evita descartar atributos relevantes sem necessidade.

A analise exploratoria tambem gera rankings de apoio:

- top 10 features por diferenca de medias entre benigno e maligno;
- top 5 features por correlacao com o alvo;
- distribuicao das classes;
- estatisticas descritivas;
- graficos em `Backend/graficos/`.

O pre-processamento aplicado no treinamento e:

- validacao de ausencia de valores nulos;
- separacao treino/teste com `stratify`;
- padronizacao com `StandardScaler`;
- treinamento com `GaussianNB`.

Ultima execucao validada:

- Acuracia: `0.9211`
- Precisao macro: `0.9215`
- Recall macro: `0.9077`
- F1 macro: `0.9138`

## Validacoes

Executado com sucesso:

- `cd Backend; .\.venv\Scripts\python.exe -m pytest`
- `cd Frontend; npm run build`
- `cd Frontend; npm run lint`

O lint do frontend termina com avisos herdados dos componentes UI sobre Fast Refresh, sem erros.
