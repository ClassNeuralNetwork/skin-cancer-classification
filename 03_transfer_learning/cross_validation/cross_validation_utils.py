"""Infraestrutura compartilhada para validação cruzada das arquiteturas."""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers, models, regularizers
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


MODULE_DIR = Path(__file__).resolve().parent
DEEP_SMOTE_DIR = MODULE_DIR.parent / "deep_smote"
if str(DEEP_SMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_SMOTE_DIR))

from project_paths import PROJECT_ROOT, resolve_project_path  # noqa: E402


CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_CODE = {label: index for index, label in enumerate(CLASSES)}
CLASS_NAME = {index: label for label, index in CLASS_CODE.items()}


@dataclass(frozen=True)
class CrossValidationConfig:
    architecture: str
    image_size: tuple[int, int]
    batch_size: int
    epochs: int = 50
    n_splits: int = 5
    learning_rate: float = 1e-4
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    seed: int = 42
    balance_strategy: str = "random_oversampling"
    oversampling_target_count: int | None = None
    backbone_weights: str | None = "imagenet"
    save_fold_models: bool = False

    @property
    def slug(self) -> str:
        return self.architecture.lower().replace("_", "-")


def configure_runtime(seed: int) -> list[tf.config.PhysicalDevice]:
    """Configura sementes e crescimento de memória da GPU."""
    keras.utils.set_random_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    return gpus


def _resolve_dataframe_paths(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for column in ["image_path", "mask_path"]:
        if column not in dataframe.columns:
            continue
        dataframe[column] = dataframe[column].map(
            lambda value: (
                value if pd.isna(value) else resolve_project_path(value)
            )
        )

    return dataframe


def _validate_image_files(dataframe: pd.DataFrame, dataset_name: str) -> None:
    missing = [
        path
        for path in dataframe["image_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} imagens ausentes em {dataset_name}. "
            f"Exemplo: {missing[0]}"
        )


def load_real_training_dataframe() -> pd.DataFrame:
    """Carrega apenas o treino real e preserva o teste final isolado."""
    deepsmote_dir = PROJECT_ROOT / "data" / "deepsmote"
    train_csv = deepsmote_dir / "train_real_split.csv"
    test_csv = deepsmote_dir / "test_real_split.csv"

    for path in [train_csv, test_csv]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}. Execute primeiro o gerador "
                "DeepSMOTE para criar as divisões reais."
            )

    train_df = _resolve_dataframe_paths(pd.read_csv(train_csv))
    test_df = _resolve_dataframe_paths(pd.read_csv(test_csv))

    required_columns = {"image", "image_path", "class_name", "label"}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        raise ValueError(
            "Colunas ausentes em train_real_split.csv: "
            + ", ".join(sorted(missing_columns))
        )

    _validate_image_files(train_df, "train_real_split.csv")
    _validate_image_files(test_df, "test_real_split.csv")

    duplicated = train_df["image"].duplicated(keep=False)
    if duplicated.any():
        examples = train_df.loc[duplicated, "image"].head(3).tolist()
        raise ValueError(f"Imagens duplicadas no treino: {examples}")

    overlap = set(train_df["image"]).intersection(test_df["image"])
    if overlap:
        raise ValueError(
            f"Há {len(overlap)} imagens presentes em treino e teste."
        )

    expected_labels = train_df["class_name"].map(CLASS_CODE)
    if not np.array_equal(
        expected_labels.to_numpy(dtype="int32"),
        train_df["label"].to_numpy(dtype="int32"),
    ):
        raise ValueError("Os códigos numéricos não correspondem às classes.")

    return train_df.reset_index(drop=True)


def build_fold_assignments(
    dataframe: pd.DataFrame,
    config: CrossValidationConfig,
) -> pd.DataFrame:
    """Cria uma atribuição determinística e estratificada de folds."""
    if config.n_splits < 2:
        raise ValueError("n_splits deve ser pelo menos 2.")

    smallest_class = int(dataframe["label"].value_counts().min())
    if smallest_class < config.n_splits:
        raise ValueError(
            "A menor classe possui menos exemplos que o número de folds."
        )

    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    assignments = dataframe[
        ["image", "image_path", "class_name", "label"]
    ].copy()
    assignments["fold"] = -1

    for fold, (_, validation_indices) in enumerate(
        splitter.split(dataframe, dataframe["label"]),
        start=1,
    ):
        assignments.loc[validation_indices, "fold"] = fold

    if (assignments["fold"] < 1).any():
        raise RuntimeError("Nem todas as imagens receberam um fold.")

    return assignments


def fold_distribution(assignments: pd.DataFrame) -> pd.DataFrame:
    """Retorna a distribuição de classes por fold para inspeção."""
    return (
        assignments.groupby(["fold", "class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
    )


def balance_training_dataframe(
    dataframe: pd.DataFrame,
    config: CrossValidationConfig,
    *,
    fold: int,
) -> pd.DataFrame:
    """Aplica oversampling aleatório somente ao treino de um fold.

    Todas as amostras originais são preservadas. As classes minoritárias
    recebem cópias escolhidas com reposição até atingirem o tamanho da maior
    classe (ou ``oversampling_target_count``, quando informado).
    """
    strategy = config.balance_strategy.lower()
    if strategy == "none":
        return dataframe.copy().reset_index(drop=True)
    if strategy != "random_oversampling":
        raise ValueError(
            "balance_strategy inválida. Use 'random_oversampling' ou 'none'."
        )

    counts = dataframe["label"].value_counts().reindex(
        range(len(CLASSES)),
        fill_value=0,
    )
    if (counts == 0).any():
        missing = [CLASS_NAME[index] for index in counts[counts == 0].index]
        raise ValueError(
            "O treino do fold não contém todas as classes: "
            + ", ".join(missing)
        )

    largest_class = int(counts.max())
    target_count = (
        largest_class
        if config.oversampling_target_count is None
        else int(config.oversampling_target_count)
    )
    if target_count < largest_class:
        raise ValueError(
            "oversampling_target_count não pode ser menor que a maior classe "
            f"do fold ({largest_class})."
        )

    balanced_parts = []
    for label in range(len(CLASSES)):
        class_rows = dataframe.loc[dataframe["label"] == label].copy()
        missing_count = target_count - len(class_rows)
        if missing_count:
            duplicated_rows = class_rows.sample(
                n=missing_count,
                replace=True,
                random_state=config.seed + fold * 100 + label,
            )
            class_rows = pd.concat(
                [class_rows, duplicated_rows],
                ignore_index=True,
            )
        balanced_parts.append(class_rows)

    return (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=config.seed + fold)
        .reset_index(drop=True)
    )


def _distribution_records(
    dataframe: pd.DataFrame,
    *,
    fold: int,
    split: str,
) -> list[dict[str, object]]:
    counts = dataframe["label"].value_counts().reindex(
        range(len(CLASSES)),
        fill_value=0,
    )
    return [
        {
            "fold": fold,
            "split": split,
            "class_name": CLASS_NAME[label],
            "label": label,
            "count": int(count),
        }
        for label, count in counts.items()
    ]


def _load_image(image_path: tf.Tensor, label: tf.Tensor, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(
    dataframe: pd.DataFrame,
    config: CrossValidationConfig,
    *,
    training: bool,
) -> tf.data.Dataset:
    paths = dataframe["image_path"].astype(str).to_numpy()
    labels = dataframe["label"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(dataframe), 4096),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: _load_image(
            path,
            label,
            config.image_size,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def _backbone_and_preprocessing(config: CrossValidationConfig):
    input_shape = (*config.image_size, 3)
    architecture = config.architecture.lower()

    if architecture == "densenet169":
        backbone = keras.applications.DenseNet169(
            weights=config.backbone_weights,
            include_top=False,
            input_shape=input_shape,
        )
        preprocess = keras.applications.densenet.preprocess_input
    elif architecture == "densenet201":
        backbone = keras.applications.DenseNet201(
            weights=config.backbone_weights,
            include_top=False,
            input_shape=input_shape,
        )
        preprocess = keras.applications.densenet.preprocess_input
    elif architecture == "efficientnetb4":
        backbone = keras.applications.EfficientNetB4(
            weights=config.backbone_weights,
            include_top=False,
            input_shape=input_shape,
        )
        preprocess = keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(
            "Arquitetura inválida. Use densenet169, densenet201 ou "
            "efficientnetb4."
        )

    return backbone, preprocess


def build_classifier(config: CrossValidationConfig) -> keras.Model:
    """Monta o classificador com a mesma cabeça para as três arquiteturas."""
    inputs = layers.Input(shape=(*config.image_size, 3), name="image")
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.seed),
            layers.RandomRotation(0.08, seed=config.seed + 1),
            layers.RandomZoom(0.08, seed=config.seed + 2),
            layers.RandomContrast(0.08, seed=config.seed + 3),
        ],
        name="augmentation",
    )

    backbone, preprocess = _backbone_and_preprocessing(config)
    backbone.trainable = False

    x = augmentation(inputs)
    x = preprocess(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = models.Model(
        inputs,
        outputs,
        name=f"{config.slug}_cross_validation",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def _calculate_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
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

    return {key: float(value) for key, value in metrics.items()}


def _save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
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
    axes[0].set_title("Matriz de confusão OOF — absoluta")
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
    axes[1].set_title("Matriz de confusão OOF — percentual")
    axes[1].set_xlabel("Classe predita")
    axes[1].set_ylabel("Classe real")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_cross_validation(config: CrossValidationConfig) -> dict[str, object]:
    """Executa todos os folds e salva métricas e predições OOF."""
    gpus = configure_runtime(config.seed)
    dataframe = load_real_training_dataframe()
    assignments = build_fold_assignments(dataframe, config)

    output_dir = (
        PROJECT_ROOT / "results" / "cross_validation" / config.slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "fold_assignments.csv", index=False)

    config_payload = asdict(config)
    config_payload["image_size"] = list(config.image_size)
    config_payload["gpu_devices"] = [gpu.name for gpu in gpus]
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    model_dir = PROJECT_ROOT / "models" / "cross_validation" / config.slug
    if config.save_fold_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    prediction_frames = []
    distribution_records = []

    for fold in range(1, config.n_splits + 1):
        print(f"\n{'=' * 72}\nFold {fold}/{config.n_splits}\n{'=' * 72}")
        keras.backend.clear_session()
        keras.utils.set_random_seed(config.seed + fold)

        validation_mask = assignments["fold"] == fold
        raw_train_df = assignments.loc[~validation_mask].reset_index(drop=True)
        validation_df = assignments.loc[validation_mask].reset_index(drop=True)

        overlap = set(raw_train_df["image"]).intersection(
            validation_df["image"]
        )
        if overlap:
            raise RuntimeError(f"Vazamento detectado no fold {fold}.")

        train_df = balance_training_dataframe(
            raw_train_df,
            config,
            fold=fold,
        )
        distribution_records.extend(
            _distribution_records(
                raw_train_df,
                fold=fold,
                split="train_before_oversampling",
            )
        )
        distribution_records.extend(
            _distribution_records(
                train_df,
                fold=fold,
                split="train_after_oversampling",
            )
        )
        distribution_records.extend(
            _distribution_records(
                validation_df,
                fold=fold,
                split="validation_untouched",
            )
        )

        train_ds = make_dataset(train_df, config, training=True)
        validation_ds = make_dataset(validation_df, config, training=False)
        model = build_classifier(config)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=config.reduce_lr_patience,
                min_lr=1e-7,
                mode="min",
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        pd.DataFrame(history.history).to_csv(
            output_dir / f"history_fold_{fold}.csv",
            index=False,
        )

        probabilities = model.predict(validation_ds, verbose=1)
        y_true = validation_df["label"].to_numpy(dtype="int32")
        y_pred = np.argmax(probabilities, axis=1).astype("int32")

        metrics = _calculate_metrics(y_true, y_pred, probabilities)
        metrics["fold"] = fold
        metrics["epochs_trained"] = len(history.history["loss"])
        metrics["training_samples_before_oversampling"] = len(raw_train_df)
        metrics["training_samples_after_oversampling"] = len(train_df)
        metrics["validation_samples"] = len(validation_df)
        fold_metrics.append(metrics)

        predictions = validation_df[
            ["image", "image_path", "class_name", "label", "fold"]
        ].copy()
        predictions["predicted_label"] = y_pred
        predictions["predicted_class"] = [CLASS_NAME[value] for value in y_pred]
        for index, class_label in enumerate(CLASSES):
            predictions[f"prob_{class_label}"] = probabilities[:, index]
        prediction_frames.append(predictions)

        if config.save_fold_models:
            model.save(model_dir / f"fold_{fold}.keras")

        del model, train_ds, validation_ds, probabilities
        gc.collect()

    metrics_df = pd.DataFrame(fold_metrics).sort_values("fold")
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    distribution_df = pd.DataFrame(distribution_records)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    distribution_df.to_csv(
        output_dir / "fold_class_distribution.csv",
        index=False,
    )

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
    ]
    summary_df = metrics_df[metric_columns].agg(["mean", "std"]).T
    summary_df.index.name = "metric"
    summary_df.to_csv(output_dir / "summary_metrics.csv")

    y_true = predictions_df["label"].to_numpy(dtype="int32")
    y_pred = predictions_df["predicted_label"].to_numpy(dtype="int32")
    y_prob = predictions_df[
        [f"prob_{class_label}" for class_label in CLASSES]
    ].to_numpy(dtype="float32")
    oof_metrics = _calculate_metrics(y_true, y_pred, y_prob)
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(oof_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _save_confusion_matrix(
        y_true,
        y_pred,
        output_dir / "oof_confusion_matrix.png",
    )

    print("\nMétricas por fold:")
    print(metrics_df.to_string(index=False))
    print("\nResumo (média ± desvio padrão):")
    print(summary_df)
    print("\nMétricas OOF:")
    print(json.dumps(oof_metrics, indent=2, ensure_ascii=False))
    print("\nResultados salvos em:", output_dir)

    return {
        "config": config,
        "fold_assignments": assignments,
        "fold_metrics": metrics_df,
        "summary_metrics": summary_df,
        "oof_predictions": predictions_df,
        "oof_metrics": oof_metrics,
        "fold_class_distribution": distribution_df,
        "output_dir": output_dir,
    }
