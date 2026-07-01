from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CLASS_LABELS, load_dataset, validate_dataset


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_PATH = MODEL_DIR / "naive_bayes_wdbc.joblib"
FEATURES_PATH = MODEL_DIR / "features.json"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
EXPERIMENTS_PATH = REPORT_DIR / "model_experiments.json"
DISCOVERY_PATH = REPORT_DIR / "descoberta_cientifica.md"

LABELS = ["B", "M"]


def make_nb_pipeline(calibrated: bool = False) -> Pipeline:
    classifier: Any
    if calibrated:
        classifier = CalibratedClassifierCV(
            estimator=GaussianNB(),
            method="sigmoid",
            cv=5,
        )
    else:
        classifier = GaussianNB()

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def select_k_best_features(x_train: pd.DataFrame, y_train: pd.Series, k: int = 10) -> list[str]:
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(x_train, y_train)
    scores = pd.Series(selector.scores_, index=x_train.columns).sort_values(ascending=False)
    return scores.head(k).index.tolist()


def select_low_correlation_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 12,
    max_abs_corr: float = 0.9,
) -> list[str]:
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(x_train, y_train)
    ranked = pd.Series(selector.scores_, index=x_train.columns).sort_values(ascending=False)
    corr = x_train.corr().abs()

    selected: list[str] = []
    for feature in ranked.index:
        if all(corr.loc[feature, chosen] < max_abs_corr for chosen in selected):
            selected.append(feature)
        if len(selected) == max_features:
            break

    return selected


def evaluate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    features: list[str],
    approach_id: str,
    name: str,
    description: str,
) -> dict[str, Any]:
    model.fit(x_train[features], y_train)
    y_pred = model.predict(x_test[features])
    y_proba = model.predict_proba(x_test[features])
    malignant_index = list(model.classes_).index("M")
    malignant_probability = y_proba[:, malignant_index]
    y_binary = (y_test == "M").astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model,
        pd.concat([x_train, x_test])[features],
        pd.concat([y_train, y_test]),
        cv=cv,
        scoring=["accuracy", "f1_macro", "recall_macro"],
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        target_names=[CLASS_LABELS[label] for label in LABELS],
        output_dict=True,
        zero_division=0,
    )
    matrix = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=LABELS),
        index=[f"real_{CLASS_LABELS[label]}" for label in LABELS],
        columns=[f"previsto_{CLASS_LABELS[label]}" for label in LABELS],
    ).to_dict()

    error_confidences = [
        float(max(probabilities))
        for probabilities, actual, predicted in zip(y_proba, y_test, y_pred)
        if actual != predicted
    ]

    return {
        "id": approach_id,
        "name": name,
        "description": description,
        "algorithm": "Gaussian Naive Bayes",
        "keeps_pdf_requirement": True,
        "features_count": len(features),
        "features": features,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, labels=LABELS, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, labels=LABELS, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, labels=LABELS, average="macro"),
        "recall_malignant": report["Tumor Maligno"]["recall"],
        "precision_malignant": report["Tumor Maligno"]["precision"],
        "brier_malignant": brier_score_loss(y_binary, malignant_probability),
        "log_loss": log_loss(y_test, y_proba, labels=LABELS),
        "mean_error_confidence": float(np.mean(error_confidences)) if error_confidences else 0.0,
        "max_error_confidence": float(np.max(error_confidences)) if error_confidences else 0.0,
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_f1_macro_mean": float(cv_scores["test_f1_macro"].mean()),
        "cv_recall_macro_mean": float(cv_scores["test_recall_macro"].mean()),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def choose_recommended(experiments: list[dict[str, Any]]) -> str:
    ranked = sorted(
        experiments,
        key=lambda item: (
            item["f1_macro"],
            item["recall_malignant"],
            -item["brier_malignant"],
            -item["mean_error_confidence"],
        ),
        reverse=True,
    )
    calibrated = [item for item in ranked if "calibrado" in item["id"]]
    if calibrated and calibrated[0]["f1_macro"] >= ranked[0]["f1_macro"] - 0.02:
        return calibrated[0]["id"]
    return ranked[0]["id"]


def write_discovery_report(experiments: list[dict[str, Any]], recommended_id: str) -> None:
    by_id = {item["id"]: item for item in experiments}
    recommended = by_id[recommended_id]

    lines = [
        "# Pipeline de Descoberta Cientifica",
        "",
        "## Restricao do trabalho",
        "",
        "O PDF exige o uso do algoritmo Naive Bayes. Por isso, todos os experimentos abaixo mantem Naive Bayes como classificador principal. As melhorias foram feitas em selecao de features, reducao de multicolinearidade e calibracao de probabilidades.",
        "",
        "## Hipotese inicial",
        "",
        "Comecamos com `GaussianNB` usando as 30 features do dataset WDBC. Essa abordagem aproveita toda a informacao da base e atende diretamente ao enunciado.",
        "",
        "## Problema observado",
        "",
        "O Naive Bayes assume que as variaveis sao independentes. No WDBC, varias features sao fortemente correlacionadas, especialmente raio, perimetro e area em diferentes agregacoes. Isso pode fazer o modelo contar evidencias parecidas mais de uma vez e gerar probabilidades muito extremas, inclusive em erros.",
        "",
        "## Experimentos executados",
        "",
    ]

    for item in experiments:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                item["description"],
                "",
                f"- Features usadas: {item['features_count']}",
                f"- Acuracia: {item['accuracy']:.4f}",
                f"- F1 macro: {item['f1_macro']:.4f}",
                f"- Recall maligno: {item['recall_malignant']:.4f}",
                f"- Brier maligno: {item['brier_malignant']:.4f}",
                f"- Confianca media nos erros: {item['mean_error_confidence']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Criterio de escolha",
            "",
            "A escolha nao olhou apenas para acuracia. Priorizamos F1 macro, recall da classe maligna e probabilidades menos exageradas, medidas por Brier score e confianca media nos erros.",
            "",
            "## Abordagem recomendada",
            "",
            f"A abordagem recomendada foi `{recommended['name']}` (`{recommended_id}`).",
            "",
            "Motivo:",
            "",
            f"- Mantem Naive Bayes, portanto respeita o PDF.",
            f"- Usa {recommended['features_count']} features.",
            f"- F1 macro: {recommended['f1_macro']:.4f}.",
            f"- Recall maligno: {recommended['recall_malignant']:.4f}.",
            f"- Brier maligno: {recommended['brier_malignant']:.4f}.",
            "",
            "## Conclusao",
            "",
            "O modelo final nao representa uma verdade clinica. Ele e uma demonstracao academica de classificacao supervisionada com Naive Bayes. A calibracao e a selecao de features ajudam a tornar a probabilidade exibida mais interpretavel, mas falsos positivos e falsos negativos ainda podem ocorrer.",
            "",
        ]
    )

    DISCOVERY_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_model() -> dict[str, Any]:
    bundle = load_dataset(cache=True)
    validate_dataset(bundle.features, bundle.target)

    x_train, x_test, y_train, y_test = train_test_split(
        bundle.features,
        bundle.target,
        test_size=0.2,
        random_state=42,
        stratify=bundle.target,
    )

    top10_features = select_k_best_features(x_train, y_train, k=10)
    low_corr_features = select_low_correlation_features(x_train, y_train, max_features=12)

    variants = [
        {
            "id": "baseline_30_features",
            "name": "Baseline - 30 features",
            "description": "Primeiro experimento: GaussianNB com todas as 30 features do WDBC.",
            "features": bundle.features.columns.tolist(),
            "model": make_nb_pipeline(calibrated=False),
        },
        {
            "id": "selectkbest_10_features",
            "name": "Selecao estatistica - 10 features",
            "description": "GaussianNB usando as 10 features com maior score ANOVA no conjunto de treino.",
            "features": top10_features,
            "model": make_nb_pipeline(calibrated=False),
        },
        {
            "id": "low_corr_12_features",
            "name": "Baixa correlacao - 12 features",
            "description": "GaussianNB com features ranqueadas por ANOVA, mas evitando pares com correlacao absoluta acima de 0.90.",
            "features": low_corr_features,
            "model": make_nb_pipeline(calibrated=False),
        },
        {
            "id": "calibrado_low_corr_12_features",
            "name": "Calibrado + baixa correlacao",
            "description": "GaussianNB com as features de menor redundancia e calibracao sigmoid para probabilidades menos extremas.",
            "features": low_corr_features,
            "model": make_nb_pipeline(calibrated=True),
        },
    ]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = []
    registry_approaches = []
    for variant in variants:
        metrics = evaluate_model(
            variant["model"],
            x_train,
            x_test,
            y_train,
            y_test,
            variant["features"],
            variant["id"],
            variant["name"],
            variant["description"],
        )
        model_path = MODEL_DIR / f"{variant['id']}.joblib"
        joblib.dump(variant["model"], model_path)
        experiments.append(metrics)
        registry_approaches.append(
            {
                "id": variant["id"],
                "name": variant["name"],
                "description": variant["description"],
                "model_path": model_path.name,
                "features": variant["features"],
                "features_count": len(variant["features"]),
                "metrics": {
                    key: metrics[key]
                    for key in [
                        "accuracy",
                        "precision_macro",
                        "recall_macro",
                        "f1_macro",
                        "recall_malignant",
                        "precision_malignant",
                        "brier_malignant",
                        "mean_error_confidence",
                    ]
                },
            }
        )

    recommended_id = choose_recommended(experiments)
    recommended = next(item for item in registry_approaches if item["id"] == recommended_id)
    recommended_model_path = MODEL_DIR / recommended["model_path"]
    joblib.dump(joblib.load(recommended_model_path), MODEL_PATH)

    registry = {
        "default_approach_id": recommended_id,
        "target_column": "Diagnosis",
        "class_labels": CLASS_LABELS,
        "all_features": bundle.features.columns.tolist(),
        "approaches": registry_approaches,
    }

    FEATURES_PATH.write_text(
        json.dumps(
            {
                "features": recommended["features"],
                "all_features": bundle.features.columns.tolist(),
                "class_labels": CLASS_LABELS,
                "target_column": "Diagnosis",
                "default_approach_id": recommended_id,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    EXPERIMENTS_PATH.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(next(item for item in experiments if item["id"] == recommended_id), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_discovery_report(experiments, recommended_id)

    return {
        "default_approach_id": recommended_id,
        "experiments": experiments,
    }


if __name__ == "__main__":
    result = train_model()
    default = next(item for item in result["experiments"] if item["id"] == result["default_approach_id"])
    print("Experimentos Naive Bayes treinados e salvos em models/")
    print(f"Abordagem recomendada: {default['name']} ({default['id']})")
    print(f"Acuracia: {default['accuracy']:.4f}")
    print(f"Precisao macro: {default['precision_macro']:.4f}")
    print(f"Recall macro: {default['recall_macro']:.4f}")
    print(f"F1 macro: {default['f1_macro']:.4f}")
