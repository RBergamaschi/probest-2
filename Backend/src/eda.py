from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

from src.data import CLASS_LABELS, TARGET_COLUMN, build_dataframe, load_dataset, validate_dataset


GRAPH_DIR = Path("graficos")
REPORT_DIR = Path("reports")
CLASS_COLORS = {"B": "#4C72B0", "M": "#C44E52"}


def save_figure(name: str) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[salvo] {path}")


def top_features_by_mean_difference(df: pd.DataFrame, n: int = 10) -> list[str]:
    features = [column for column in df.columns if column != TARGET_COLUMN]
    means = df.groupby(TARGET_COLUMN)[features].mean()
    return (means.loc["M"] - means.loc["B"]).abs().nlargest(n).index.tolist()


def top_features_by_target_correlation(df: pd.DataFrame, n: int = 5) -> list[str]:
    numeric_target = df[TARGET_COLUMN].map({"B": 0, "M": 1})
    features = [column for column in df.columns if column != TARGET_COLUMN]
    return df[features].corrwith(numeric_target).abs().nlargest(n).index.tolist()


def generate_eda() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 120})

    bundle = load_dataset(cache=True)
    validate_dataset(bundle.features, bundle.target)
    df = build_dataframe(bundle)
    features = [column for column in df.columns if column != TARGET_COLUMN]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.describe().T.to_csv(REPORT_DIR / "estatisticas_descritivas.csv")
    df.groupby(TARGET_COLUMN)[features].agg(["mean", "std"]).to_csv(
        REPORT_DIR / "estatisticas_por_classe.csv"
    )

    class_counts = df[TARGET_COLUMN].value_counts().reindex(["B", "M"])
    class_summary = pd.DataFrame(
        {
            "classe": [CLASS_LABELS[label] for label in class_counts.index],
            "quantidade": class_counts.values,
            "percentual": (class_counts.values / len(df) * 100).round(2),
        },
        index=class_counts.index,
    )
    class_summary.to_csv(REPORT_DIR / "distribuicao_classes.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [CLASS_LABELS[label] for label in class_counts.index]
    colors = [CLASS_COLORS[label] for label in class_counts.index]
    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Contagem Absoluta")
    axes[0].set_xlabel("Classe do Tumor")
    axes[0].set_ylabel("Numero de Casos")
    for index, value in enumerate(class_counts.values):
        axes[0].text(index, value + 5, str(value), ha="center", fontweight="bold")
    axes[1].pie(
        class_counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Proporcao (%)")
    fig.suptitle("Distribuicao das Classes - Benigno vs Maligno", fontweight="bold")
    save_figure("01_distribuicao_classes")

    n_cols = 5
    n_rows = -(-len(features) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.2))
    axes_flat = axes.flatten()
    for index, feature in enumerate(features):
        ax = axes_flat[index]
        for label in ["B", "M"]:
            ax.hist(
                df.loc[df[TARGET_COLUMN] == label, feature],
                bins=25,
                alpha=0.55,
                color=CLASS_COLORS[label],
                density=True,
                label=CLASS_LABELS[label],
            )
        ax.set_title(feature, fontsize=8)
        ax.tick_params(labelsize=7)
    for index in range(len(features), len(axes_flat)):
        axes_flat[index].set_visible(False)
    fig.legend(
        handles=[mpatches.Patch(color=CLASS_COLORS[label], label=CLASS_LABELS[label]) for label in ["B", "M"]],
        loc="upper right",
    )
    fig.suptitle("Distribuicao das Variaveis Preditoras por Classe", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure("02_histogramas_features")

    top10 = top_features_by_mean_difference(df, n=10)
    fig, axes = plt.subplots(5, 2, figsize=(14, 17))
    axes_flat = axes.flatten()
    for index, feature in enumerate(top10):
        ax = axes_flat[index]
        box_data = [df.loc[df[TARGET_COLUMN] == "B", feature], df.loc[df[TARGET_COLUMN] == "M", feature]]
        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )
        for patch, label in zip(bp["boxes"], ["B", "M"]):
            patch.set_facecolor(CLASS_COLORS[label])
            patch.set_alpha(0.7)
        ax.set_title(feature, fontsize=9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([CLASS_LABELS["B"], CLASS_LABELS["M"]])
    fig.suptitle("Boxplots por Classe - Top 10 Features por Diferenca de Medias", fontweight="bold")
    plt.tight_layout()
    save_figure("03_boxplots_top10_features")

    numeric_df = df[features].copy()
    numeric_df["target_maligno"] = df[TARGET_COLUMN].map({"B": 0, "M": 1})
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        numeric_df.corr(),
        ax=ax,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.3,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Correlacao de Pearson"},
    )
    ax.set_title("Heatmap de Correlacao de Pearson - Features e Target", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    save_figure("04_heatmap_correlacao")

    top5 = top_features_by_target_correlation(df, n=5)
    plot_df = df[top5 + [TARGET_COLUMN]].copy()
    plot_df["Classe"] = plot_df[TARGET_COLUMN].map(CLASS_LABELS)
    grid = sns.pairplot(
        plot_df.drop(columns=[TARGET_COLUMN]),
        hue="Classe",
        palette={CLASS_LABELS[label]: CLASS_COLORS[label] for label in ["B", "M"]},
        diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 18},
    )
    grid.figure.suptitle("Pairplot - 5 Features Mais Correlacionadas com Diagnostico", y=1.02)
    save_figure("05_pairplot_top5_features")

    summary = {
        "dataset": "Breast Cancer Wisconsin Diagnostic Dataset",
        "source": bundle.source,
        "observacoes": len(df),
        "features": len(features),
        "valores_nulos": int(df.isna().sum().sum()),
        "classes": class_summary.to_dict(orient="index"),
        "top10_diferenca_medias": top10,
        "top5_correlacao_target": top5,
    }
    pd.Series(summary, dtype="object").to_json(REPORT_DIR / "eda_summary.json", indent=2)
    print("EDA concluida.")


if __name__ == "__main__":
    generate_eda()
