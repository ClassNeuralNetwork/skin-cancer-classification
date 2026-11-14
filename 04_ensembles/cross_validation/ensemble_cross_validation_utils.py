"""Funções para avaliar o ensemble a partir das predições out-of-fold."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "cross_validation"

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAME = {index: name for index, name in enumerate(CLASSES)}
IDENTITY_COLUMNS = ["image", "class_name", "label", "fold"]
PROBABILITY_COLUMNS = [f"prob_{class_name}" for class_name in CLASSES]


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(CLASSES)),
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    return {name: float(value) for name, value in metrics.items()}


def _load_oof_predictions(model_slug: str) -> pd.DataFrame:
    predictions_path = RESULTS_ROOT / model_slug / "oof_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Predições não encontradas para {model_slug}: "
            f"{predictions_path}"
        )

    dataframe = pd.read_csv(predictions_path)
    required_columns = set(IDENTITY_COLUMNS + PROBABILITY_COLUMNS)
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{model_slug} não contém as colunas: "
            + ", ".join(sorted(missing_columns))
        )
    if dataframe["image"].duplicated().any():
        raise ValueError(
            f"Há imagens repetidas nas predições de {model_slug}."
        )

    return dataframe.sort_values("image").reset_index(drop=True)


def load_aligned_predictions(
    model_slugs: list[str],
) -> dict[str, pd.DataFrame]:
    """Carrega as predições e confirma que os modelos usaram os mesmos folds."""
    if len(model_slugs) < 2:
        raise ValueError("O ensemble precisa de pelo menos dois modelos.")

    predictions = {
        model_slug: _load_oof_predictions(model_slug)
        for model_slug in model_slugs
    }
    reference_slug = model_slugs[0]
    reference = predictions[reference_slug][IDENTITY_COLUMNS]

    for model_slug in model_slugs[1:]:
        candidate = predictions[model_slug][IDENTITY_COLUMNS]
        if not reference.equals(candidate):
            raise ValueError(
                f"Os folds ou rótulos de {model_slug} não coincidem com "
                f"{reference_slug}."
            )

    return predictions


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype="float64"),
        where=row_totals != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[0],
    )
    axes[0].set_title("Matriz de confusão OOF — valores absolutos")
    axes[0].set_xlabel("Classe predita")
    axes[0].set_ylabel("Classe real")

    sns.heatmap(
        normalized * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[1],
    )
    axes[1].set_title("Matriz de confusão OOF — percentual por classe")
    axes[1].set_xlabel("Classe predita")
    axes[1].set_ylabel("Classe real")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_soft_voting_ensemble(
    model_slugs: list[str],
    *,
    weights: list[float] | None = None,
    ensemble_slug: str = "ensemble-3-soft-voting",
) -> dict[str, object]:
    """Combina probabilidades OOF com pesos definidos antes da avaliação."""
    predictions = load_aligned_predictions(model_slugs)

    if weights is None:
        weights_array = np.ones(len(model_slugs), dtype="float64")
    else:
        weights_array = np.asarray(weights, dtype="float64")
    if weights_array.shape != (len(model_slugs),):
        raise ValueError("Informe um peso para cada modelo do ensemble.")
    if not np.isfinite(weights_array).all() or (weights_array < 0).any():
        raise ValueError("Os pesos devem ser números finitos e não negativos.")
    if weights_array.sum() == 0:
        raise ValueError("Pelo menos um peso deve ser maior que zero.")
    weights_array = weights_array / weights_array.sum()

    reference = predictions[model_slugs[0]].copy()
    probability_stack = np.stack(
        [
            predictions[model_slug][PROBABILITY_COLUMNS].to_numpy(
                dtype="float64"
            )
            for model_slug in model_slugs
        ],
        axis=0,
    )
    ensemble_probabilities = np.average(
        probability_stack,
        axis=0,
        weights=weights_array,
    )
    ensemble_labels = np.argmax(ensemble_probabilities, axis=1).astype(
        "int32"
    )

    output = reference[IDENTITY_COLUMNS].copy()
    if "image_path" in reference.columns:
        output.insert(1, "image_path", reference["image_path"])
    output["predicted_label"] = ensemble_labels
    output["predicted_class"] = [
        CLASS_NAME[label] for label in ensemble_labels
    ]
    for index, class_name in enumerate(CLASSES):
        output[f"prob_{class_name}"] = ensemble_probabilities[:, index]

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
    ]
    fold_metrics = []
    for fold in sorted(output["fold"].unique()):
        fold_mask = output["fold"] == fold
        fold_true = output.loc[fold_mask, "label"].to_numpy(dtype="int32")
        fold_pred = output.loc[
            fold_mask,
            "predicted_label",
        ].to_numpy(dtype="int32")
        fold_prob = output.loc[
            fold_mask,
            PROBABILITY_COLUMNS,
        ].to_numpy(dtype="float64")
        metrics = _calculate_metrics(fold_true, fold_pred, fold_prob)
        metrics["fold"] = int(fold)
        metrics["validation_samples"] = int(fold_mask.sum())
        fold_metrics.append(metrics)

    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values("fold")
    summary_df = fold_metrics_df[metric_names].agg(["mean", "std"]).T
    summary_df.index.name = "metric"

    y_true = output["label"].to_numpy(dtype="int32")
    y_pred = output["predicted_label"].to_numpy(dtype="int32")
    y_prob = output[PROBABILITY_COLUMNS].to_numpy(dtype="float64")
    oof_metrics = _calculate_metrics(y_true, y_pred, y_prob)

    comparison_rows = []
    for model_slug in model_slugs:
        model_predictions = predictions[model_slug]
        model_true = model_predictions["label"].to_numpy(dtype="int32")
        model_pred = model_predictions["predicted_label"].to_numpy(
            dtype="int32"
        )
        model_prob = model_predictions[PROBABILITY_COLUMNS].to_numpy(
            dtype="float64"
        )
        comparison_rows.append(
            {"model": model_slug, **_calculate_metrics(
                model_true,
                model_pred,
                model_prob,
            )}
        )
    comparison_rows.append({"model": ensemble_slug, **oof_metrics})
    comparison_df = pd.DataFrame(comparison_rows)

    output_dir = RESULTS_ROOT / ensemble_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv")
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(oof_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "method": "soft_voting",
                "model_slugs": model_slugs,
                "normalized_weights": {
                    model_slug: float(weight)
                    for model_slug, weight in zip(
                        model_slugs,
                        weights_array,
                    )
                },
                "weight_selection": "defined_before_oof_evaluation",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_confusion_matrix(
        y_true,
        y_pred,
        output_dir / "oof_confusion_matrix.png",
    )

    return {
        "oof_predictions": output,
        "fold_metrics": fold_metrics_df,
        "summary_metrics": summary_df,
        "oof_metrics": oof_metrics,
        "model_comparison": comparison_df,
        "output_dir": output_dir,
    }
