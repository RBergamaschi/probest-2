import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Força UTF-8 no terminal do Windows para caracteres especiais
sys.stdout.reconfigure(encoding="utf-8")

# ── Configurações globais de estilo ──────────────────────────────────────────
# Define paleta consistente: benigno = azul, maligno = vermelho
CORES_CLASSE = {0: "#4C72B0", 1: "#C44E52"}   # índice da classe → cor hex
NOMES_CLASSE = {0: "Benigno", 1: "Maligno"}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

# Pasta onde todos os gráficos serão salvos
PASTA_GRAFICOS = "graficos"
os.makedirs(PASTA_GRAFICOS, exist_ok=True)   # cria a pasta se não existir

SEPARADOR = "\n" + "=" * 65 + "\n"


def salvar_figura(nome_arquivo: str) -> None:
    """Salva a figura atual na pasta de gráficos e fecha o plot.

    Args:
        nome_arquivo: nome do arquivo sem extensão (ex: 'histogramas').
    """
    caminho = os.path.join(PASTA_GRAFICOS, f"{nome_arquivo}.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [salvo] {caminho}")


# =============================================================================
# BLOCO 1: CARREGAMENTO E ESTRUTURA DOS DADOS
# =============================================================================

def carregar_dados() -> pd.DataFrame:
    #Carrega o Breast Cancer Wisconsin Dataset e retorna um DataFrame Pandas.
    dataset = load_breast_cancer()

    # Monta DataFrame: cada linha é um tumor, cada coluna é uma característica
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

    # Adiciona a variável-alvo como última coluna (0 = benigno, 1 = maligno)
    df["target"] = dataset.target

    return df, dataset


print(SEPARADOR)
print("BLOCO 1 — CARREGAMENTO E ESTRUTURA DOS DADOS")
print(SEPARADOR)

df, dataset = carregar_dados()

print(f"Dimensões do DataFrame: {df.shape[0]} observações × {df.shape[1]} colunas\n")

print("Primeiras 5 linhas do dataset:")
print(df.head().to_string())

print("\nInformações gerais (tipos de dados e não-nulos):")
df.info()

print("\nEstatísticas descritivas de todas as colunas numéricas:")

print(df.describe().T.to_string())


# =============================================================================
# BLOCO 2: DISTRIBUIÇÃO DA VARIÁVEL-ALVO (CLASSES)
# =============================================================================

def analisar_classes(df: pd.DataFrame) -> None:
    #Conta e visualiza a distribuição das classes (Benigno vs Maligno).

    contagem = df["target"].value_counts().sort_index()
    rotulos = [NOMES_CLASSE[i] for i in contagem.index]
    cores = [CORES_CLASSE[i] for i in contagem.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição das Classes — Benigno vs Maligno", fontsize=15, fontweight="bold")

    # ── Gráfico de barras: contagem absoluta ──────────────────────────────
    axes[0].bar(rotulos, contagem.values, color=cores, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Contagem Absoluta")
    axes[0].set_xlabel("Classe do Tumor")
    axes[0].set_ylabel("Número de Casos")

    # Adiciona o número no topo de cada barra
    for i, (rotulo, valor) in enumerate(zip(rotulos, contagem.values)):
        axes[0].text(i, valor + 5, str(valor), ha="center", fontweight="bold", fontsize=12)

    # ── Gráfico de pizza: proporção percentual ────────────────────────────
    axes[1].pie(
        contagem.values,
        labels=rotulos,
        colors=cores,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Proporção (%)")

    salvar_figura("01_distribuicao_classes")


print(SEPARADOR)
print("BLOCO 2 — DISTRIBUIÇÃO DA VARIÁVEL-ALVO")
print(SEPARADOR)

contagem_classes = df["target"].value_counts().sort_index()
for classe, qtd in contagem_classes.items():
    pct = qtd / len(df) * 100
    print(f"  {NOMES_CLASSE[classe]:>8}: {qtd:>4} casos ({pct:.1f}%)")

analisar_classes(df)


# =============================================================================
# BLOCO 3: ESTATÍSTICAS DESCRITIVAS POR CLASSE
# =============================================================================

def estatisticas_por_classe(df: pd.DataFrame) -> None:
    """Exibe média e desvio padrão de cada feature, separado por classe.

    Permite identificar quais variáveis têm maior diferença entre benignos
    e malignos — candidatas a boas features discriminativas.
    """
    features = [c for c in df.columns if c != "target"]

    stats_benigno = df[df["target"] == 0][features].describe().T[["mean", "std"]]
    stats_maligno = df[df["target"] == 1][features].describe().T[["mean", "std"]]

    # Renomeia colunas para deixar claro de qual classe vêm
    stats_benigno.columns = ["Média Benigno", "DP Benigno"]
    stats_maligno.columns = ["Média Maligno", "DP Maligno"]

    comparativo = pd.concat([stats_benigno, stats_maligno], axis=1)

    # Diferença absoluta entre médias — útil para rankear features
    comparativo["Diferença Médias"] = abs(
        comparativo["Média Maligno"] - comparativo["Média Benigno"]
    )

    print("Estatísticas descritivas por classe (ordenadas por diferença de médias):")
    print(comparativo.sort_values("Diferença Médias", ascending=False).to_string())


print(SEPARADOR)
print("BLOCO 3 — ESTATÍSTICAS DESCRITIVAS POR CLASSE")
print(SEPARADOR)

estatisticas_por_classe(df)


# =============================================================================
# BLOCO 4: VERIFICAÇÃO DE VALORES NULOS
# =============================================================================

def verificar_nulos(df: pd.DataFrame) -> None:
    """Verifica e reporta a presença de valores ausentes no DataFrame."""
    nulos_por_coluna = df.isnull().sum()          # conta NaN por coluna
    total_nulos = nulos_por_coluna.sum()           # total geral

    if total_nulos == 0:
        print("Nenhum valor nulo encontrado. Dataset está completo.")
    else:
        colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]
        print(f"Total de valores nulos: {total_nulos}")
        print(colunas_com_nulos.to_string())


print(SEPARADOR)
print("BLOCO 4 — VERIFICAÇÃO DE VALORES NULOS")
print(SEPARADOR)

verificar_nulos(df)


# =============================================================================
# BLOCO 5: HISTOGRAMAS DAS VARIÁVEIS PREDITORAS
# =============================================================================

def plotar_histogramas(df: pd.DataFrame) -> None:
    """Plota histogramas de todas as 30 features, com KDE sobreposta.

    Cada histograma mostra a distribuição de frequência da feature,
    com as classes Benigno e Maligno sobrepostas para comparação visual.
    """
    features = [c for c in df.columns if c != "target"]
    n_cols = 5
    n_rows = -(-len(features) // n_cols)   # divisão com teto (ceiling division)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.2))
    fig.suptitle("Distribuição das Variáveis Preditoras por Classe", fontsize=16, fontweight="bold", y=1.01)

    axes_flat = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes_flat[idx]

        for classe in [0, 1]:
            valores = df[df["target"] == classe][feature]
            ax.hist(
                valores,
                bins=25,
                alpha=0.55,   
                color=CORES_CLASSE[classe],
                label=NOMES_CLASSE[classe],
                density=True,                   
            )

        ax.set_title(feature, fontsize=8, pad=3)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)

    # Esconde eixos dos subplots vazios (se n_features não preenche a grade)
    for idx in range(len(features), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Legenda única no canto superior direito da figura
    legend_patches = [
        mpatches.Patch(color=CORES_CLASSE[0], label="Benigno"),
        mpatches.Patch(color=CORES_CLASSE[1], label="Maligno"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=11)

    plt.tight_layout()
    salvar_figura("02_histogramas_features")


print(SEPARADOR)
print("BLOCO 5 — HISTOGRAMAS DAS VARIÁVEIS PREDITORAS")
print(SEPARADOR)

plotar_histogramas(df)


# =============================================================================
# BLOCO 6: BOXPLOTS POR CLASSE — TOP 10 FEATURES COM MAIOR DIFERENÇA DE MÉDIAS
# =============================================================================

def selecionar_top_features_por_diferenca(df: pd.DataFrame, n: int = 10) -> list[str]:
    """Retorna as N features com maior diferença absoluta de médias entre classes.

    Args:
        df: DataFrame com coluna 'target'.
        n: número de features a retornar.

    Returns:
        Lista com os nomes das N features mais discriminativas pela média.
    """
    features = [c for c in df.columns if c != "target"]
    media_benigno = df[df["target"] == 0][features].mean()
    media_maligno = df[df["target"] == 1][features].mean()

    diferenca = (media_maligno - media_benigno).abs()
    return diferenca.nlargest(n).index.tolist()


def plotar_boxplots(df: pd.DataFrame, features: list[str]) -> None:
    """Plota boxplots lado a lado (Benigno vs Maligno) para as features selecionadas.

    Args:
        df: DataFrame completo com coluna 'target'.
        features: lista de nomes de features a plotar.
    """
    n_cols = 2
    n_rows = -(-len(features) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    fig.suptitle(
        "Boxplots por Classe — Top 10 Features com Maior Diferença de Médias",
        fontsize=14, fontweight="bold",
    )

    axes_flat = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes_flat[idx]

        dados_boxplot = [
            df[df["target"] == 0][feature].values,   # valores para Benigno
            df[df["target"] == 1][feature].values,   # valores para Maligno
        ]

        bp = ax.boxplot(
            dados_boxplot,
            patch_artist=True,        # preenche as caixas com cor
            notch=False,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )

        # Aplica as cores das classes às caixas
        for patch, classe in zip(bp["boxes"], [0, 1]):
            patch.set_facecolor(CORES_CLASSE[classe])
            patch.set_alpha(0.7)

        ax.set_title(feature, fontsize=9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Benigno", "Maligno"])
        ax.set_ylabel("Valor")

    for idx in range(len(features), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    salvar_figura("03_boxplots_top10_features")


print(SEPARADOR)
print("BLOCO 6 — BOXPLOTS POR CLASSE (TOP 10 FEATURES)")
print(SEPARADOR)

top10_features = selecionar_top_features_por_diferenca(df, n=10)
print(f"Top 10 features com maior diferença de médias entre classes:")
for i, f in enumerate(top10_features, 1):
    print(f"  {i:>2}. {f}")

plotar_boxplots(df, top10_features)


# =============================================================================
# BLOCO 7: HEATMAP DE CORRELAÇÃO COMPLETO
# =============================================================================

def plotar_heatmap_correlacao(df: pd.DataFrame) -> None:
    """Plota heatmap de correlação de Pearson entre todas as features + target.

    Células vermelhas intensas indicam alta correlação positiva.
    Células azuis intensas indicam alta correlação negativa.
    Diagonal principal = 1.0 (cada variável consigo mesma).
    """
    # Calcula matriz de correlação de Pearson
    matriz_corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(22, 18))

    sns.heatmap(
        matriz_corr,
        ax=ax,
        cmap="coolwarm",          # vermelho = positivo, azul = negativo
        center=0,                 # âncora da paleta no zero
        annot=False,              # sem números (muitas células → ilegível)
        linewidths=0.3,
        linecolor="white",
        square=True,              # células quadradas
        cbar_kws={"shrink": 0.7, "label": "Correlação de Pearson"},
    )

    ax.set_title("Heatmap de Correlação de Pearson — Todas as Features", fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    salvar_figura("04_heatmap_correlacao")


def identificar_alta_correlacao(df: pd.DataFrame, limiar: float = 0.9) -> None:
    """Imprime pares de features com correlação acima do limiar especificado.

    Pares com |correlação| > 0.9 são candidatos à redundância (multicolinearidade).

    Args:
        df: DataFrame com as features.
        limiar: valor de corte para alertar sobre alta correlação.
    """
    features = [c for c in df.columns if c != "target"]
    matriz = df[features].corr()

    pares_alta_corr = []
    for i in range(len(matriz.columns)):
        for j in range(i + 1, len(matriz.columns)):   # triângulo superior apenas
            corr_val = matriz.iloc[i, j]
            if abs(corr_val) >= limiar:
                pares_alta_corr.append((matriz.columns[i], matriz.columns[j], corr_val))

    pares_alta_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nPares de features com correlação absoluta ≥ {limiar}:")
    if pares_alta_corr:
        for f1, f2, corr in pares_alta_corr:
            print(f"  {f1:>35}  ↔  {f2:<35}  r = {corr:+.3f}")
    else:
        print("  Nenhum par encontrado acima do limiar.")


print(SEPARADOR)
print("BLOCO 7 — HEATMAP DE CORRELAÇÃO")
print(SEPARADOR)

plotar_heatmap_correlacao(df)
identificar_alta_correlacao(df, limiar=0.9)


# =============================================================================
# BLOCO 8: PAIRPLOT DAS 5 FEATURES MAIS CORRELACIONADAS COM O TARGET
# =============================================================================

def selecionar_top_features_corr_target(df: pd.DataFrame, n: int = 5) -> list[str]:
    """Retorna as N features com maior correlação absoluta com o target.

    Args:
        df: DataFrame com coluna 'target'.
        n: número de features a retornar.

    Returns:
        Lista com os nomes das N features mais correlacionadas com o target.
    """
    features = [c for c in df.columns if c != "target"]
    corr_com_target = df[features].corrwith(df["target"]).abs()
    return corr_com_target.nlargest(n).index.tolist()


def plotar_pairplot(df: pd.DataFrame, features: list[str]) -> None:
    """Plota pairplot das features selecionadas, colorido por classe.

    Na diagonal: KDE (curva de densidade) de cada feature separada por classe.
    Fora da diagonal: scatterplot de cada par de features, com pontos coloridos
    por classe — permite avaliar a separabilidade visual.

    Args:
        df: DataFrame com coluna 'target'.
        features: lista de features a incluir no pairplot.
    """
    # Cria DataFrame auxiliar com coluna de rótulo textual (para a legenda)
    df_plot = df[features + ["target"]].copy()
    df_plot["Classe"] = df_plot["target"].map(NOMES_CLASSE)

    paleta = {NOMES_CLASSE[0]: CORES_CLASSE[0], NOMES_CLASSE[1]: CORES_CLASSE[1]}

    g = sns.pairplot(
        df_plot,
        hue="Classe",               # colore por classe
        palette=paleta,
        diag_kind="kde",            # diagonal: curva de densidade suavizada
        plot_kws={"alpha": 0.45, "s": 18},   # transparência e tamanho dos pontos
        diag_kws={"fill": True, "alpha": 0.5},
    )

    g.figure.suptitle(
        "Pairplot — 5 Features Mais Correlacionadas com o Target",
        y=1.02, fontsize=14, fontweight="bold",
    )

    salvar_figura("05_pairplot_top5_features")


print(SEPARADOR)
print("BLOCO 8 — PAIRPLOT DAS 5 FEATURES MAIS CORRELACIONADAS COM O TARGET")
print(SEPARADOR)

top5_features = selecionar_top_features_corr_target(df, n=5)
print("Top 5 features mais correlacionadas com o target (Benigno/Maligno):")
features_corr = df[[c for c in df.columns if c != "target"]].corrwith(df["target"]).abs()
for i, f in enumerate(top5_features, 1):
    print(f"  {i}. {f:<40}  |r| = {features_corr[f]:.4f}")

plotar_pairplot(df, top5_features)


# =============================================================================
# RESUMO FINAL DOS PRINCIPAIS ACHADOS
# =============================================================================

print(SEPARADOR)
print("RESUMO FINAL DA ANÁLISE EXPLORATÓRIA")
print(SEPARADOR)

n_total = len(df)
n_benigno = (df["target"] == 0).sum()
n_maligno = (df["target"] == 1).sum()
pct_benigno = n_benigno / n_total * 100
pct_maligno = n_maligno / n_total * 100

features = [c for c in df.columns if c != "target"]
n_features = len(features)
nulos_total = df.isnull().sum().sum()

# Recalcula pares de alta correlação para o resumo
matriz_corr_feat = df[features].corr()
n_pares_alta_corr = sum(
    1
    for i in range(len(features))
    for j in range(i + 1, len(features))
    if abs(matriz_corr_feat.iloc[i, j]) >= 0.9
)

# Feature com maior correlação com o target
melhor_feature = features_corr.idxmax()
melhor_corr = features_corr.max()

print(f"""
  Dataset ......... Breast Cancer Wisconsin Diagnostic Dataset
  Observações ..... {n_total}
  Features ........ {n_features} variáveis preditoras contínuas
  Valores nulos ... {nulos_total} (dataset completo)

  BALANCEAMENTO DE CLASSES
    Benigno ....... {n_benigno} casos ({pct_benigno:.1f}%)
    Maligno ....... {n_maligno} casos ({pct_maligno:.1f}%)
    → Dataset moderadamente balanceado; acurácia é métrica válida, mas
      recomenda-se também avaliar F1-Score e Recall para a classe Maligno.

  CORRELAÇÕES
    Pares com |r| ≥ 0.90 ... {n_pares_alta_corr} pares
    → Alta multicolinearidade detectada entre features de média, pior caso
      e erro padrão de mesmas medidas (raio, perímetro, área).
    → Considerar PCA ou seleção de features antes da modelagem.

  FEATURE MAIS DISCRIMINATIVA
    {melhor_feature}
    Correlação com target: |r| = {melhor_corr:.4f}

  TOP 5 FEATURES (por correlação com target)
""")

for i, f in enumerate(top5_features, 1):
    print(f"    {i}. {f:<40} |r| = {features_corr[f]:.4f}")

print(f"""
  GRÁFICOS SALVOS EM: ./{PASTA_GRAFICOS}/
    01_distribuicao_classes.png
    02_histogramas_features.png
    03_boxplots_top10_features.png
    04_heatmap_correlacao.png
    05_pairplot_top5_features.png
""")
print(SEPARADOR)
