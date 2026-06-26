import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import random
from pathlib import Path
import json
from datetime import datetime
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

possible_roots = [
    Path("."),
    Path(".."),
    Path("../..")
]

PROJECT_ROOT = None

for root in possible_roots:
    if (root / "data" / "GroundTruth.csv").exists():
        PROJECT_ROOT = root.resolve()
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Não foi possível localizar a raiz do projeto. "
        "Verifique se existe data/GroundTruth.csv."
    )

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
CSV_PATH = DATA_DIR / "GroundTruth.csv"

DEEPSMOTE_DIR = DATA_DIR / "deepsmote"
TEST_SPLIT_PATH = DEEPSMOTE_DIR / "test_real_split.csv"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results" / "ensemble_deepsmote_model5_oversampling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
class_code = {label: idx for idx, label in enumerate(classes)}
class_name = {idx: label for label, idx in class_code.items()}

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("Classes:", class_code)

# Modelo A: DenseNet201 + DeepSMOTE
IMG_SIZE_A = (320, 320)
MODEL_A_PATH = MODELS_DIR / "best_densenet201_deepsmote_saved_images.keras"

# Modelo B: DenseNet169 + DeepSMOTE
IMG_SIZE_B = (320, 320)
MODEL_B_PATH = MODELS_DIR / "best_densenet169_deepsmote_saved_images.keras"

# Modelo C: EfficientNetB4 + DeepSMOTE
IMG_SIZE_C = (380, 380)
MODEL_C_PATH = MODELS_DIR / "best_efficientnetb4_deepsmote_saved_images.keras"

USE_DULL_RAZOR = False
USE_MASK_SEGMENTATION = False
USE_MASK_CROP = True
CROP_MARGIN = 20

WEIGHT_MODEL_A = 0.70
WEIGHT_MODEL_B = 0.20
WEIGHT_MODEL_C = 0.10

print("Modelo A:", IMG_SIZE_A, MODEL_A_PATH)
print("Modelo B:", IMG_SIZE_B, MODEL_B_PATH)
print("Modelo C:", IMG_SIZE_C, MODEL_C_PATH)
print("Pesos:", WEIGHT_MODEL_A, WEIGHT_MODEL_B, WEIGHT_MODEL_C)

assert MODEL_A_PATH.exists(), f"Modelo A não encontrado: {MODEL_A_PATH}"
assert MODEL_B_PATH.exists(), f"Modelo B não encontrado: {MODEL_B_PATH}"
assert MODEL_C_PATH.exists(), f"Modelo C não encontrado: {MODEL_C_PATH}"

assert abs((WEIGHT_MODEL_A + WEIGHT_MODEL_B + WEIGHT_MODEL_C) - 1.0) < 1e-6, "Os pesos devem somar 1."

LOCAL_CROP_RATIO = 0.90
INCLUDE_FULL_IMAGE = True
USE_HORIZONTAL_FLIP = True
USE_VERTICAL_FLIP = False
USE_HV_FLIP = False

print("LOCAL_CROP_RATIO:", LOCAL_CROP_RATIO)
print("INCLUDE_FULL_IMAGE:", INCLUDE_FULL_IMAGE)
print("USE_HORIZONTAL_FLIP:", USE_HORIZONTAL_FLIP)
print("USE_VERTICAL_FLIP:", USE_VERTICAL_FLIP)
print("USE_HV_FLIP:", USE_HV_FLIP)

def find_image_path(image_id, images_dir):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = images_dir / f"{image_id}{ext}"
        if path.exists():
            return str(path)
    return None


def find_mask_path(image_id, masks_dir):
    if not masks_dir.exists():
        return None

    candidates = [
        masks_dir / f"{image_id}_segmentation.png",
        masks_dir / f"{image_id}_segmentation.jpg",
        masks_dir / f"{image_id}_mask.png",
        masks_dir / f"{image_id}_mask.jpg",
        masks_dir / f"{image_id}.png",
        masks_dir / f"{image_id}.jpg",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    matches = list(masks_dir.glob(f"{image_id}*"))
    if len(matches) > 0:
        return str(matches[0])

    return None


def build_test_df_from_groundtruth():
    metadata = pd.read_csv(CSV_PATH)

    rows = []
    for _, row in metadata.iterrows():
        image_id = str(row["image"])
        image_path = find_image_path(image_id, IMAGES_DIR)
        mask_path = find_mask_path(image_id, MASKS_DIR)

        if image_path is None:
            continue

        label_name = row[classes].idxmax()
        label = class_code[label_name]

        rows.append({
            "image": image_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "class_name": label_name,
            "label": label
        })

    df = pd.DataFrame(rows)

    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

    return test_df.reset_index(drop=True)


if TEST_SPLIT_PATH.exists():
    print("Usando split de teste salvo em:", TEST_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
else:
    print("test_real_split.csv não encontrado. Recriando split a partir do GroundTruth.csv.")
    test_df = build_test_df_from_groundtruth()

print("Teste:", len(test_df))
print("\nDistribuição do teste:")
print(test_df["class_name"].value_counts().reindex(classes))

def read_rgb_image(path):
    if isinstance(path, bytes):
        path = path.decode("utf-8")

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def dull_razor_remove_hair(image_rgb, kernel_size=(9, 9), threshold_value=10, inpaint_radius=6):
    image_rgb = image_rgb.astype(np.uint8)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat_blur = cv2.GaussianBlur(blackhat, (3, 3), 0)

    _, mask = cv2.threshold(blackhat_blur, threshold_value, 255, cv2.THRESH_BINARY)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    clean_bgr = cv2.inpaint(image_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

    return cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)


def read_binary_mask(mask_path, target_shape):
    if mask_path is None or pd.isna(mask_path):
        return None

    if isinstance(mask_path, bytes):
        mask_path = mask_path.decode("utf-8")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    height, width = target_shape[:2]
    mask = cv2.resize(mask, (width, height))

    binary_mask = (mask > 127).astype(np.uint8)

    if binary_mask.sum() == 0:
        return None

    return binary_mask


def apply_mask_segmentation(image_rgb, mask_path):
    binary_mask = read_binary_mask(mask_path, target_shape=image_rgb.shape)

    if binary_mask is None:
        return image_rgb

    segmented = image_rgb * np.expand_dims(binary_mask, axis=-1)

    if np.mean(segmented) < 5:
        return image_rgb

    return segmented


def crop_lesion_with_margin(image_rgb, mask_path, margin=20):
    binary_mask = read_binary_mask(mask_path, target_shape=image_rgb.shape)

    if binary_mask is None:
        return image_rgb

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image_rgb

    height, width = image_rgb.shape[:2]

    x_min = max(xs.min() - margin, 0)
    x_max = min(xs.max() + margin, width - 1)
    y_min = max(ys.min() - margin, 0)
    y_max = min(ys.max() + margin, height - 1)

    cropped = image_rgb[y_min:y_max + 1, x_min:x_max + 1]

    if cropped.size == 0:
        return image_rgb

    return cropped


def preprocess_base_image(path, mask_path=None):
    image_rgb = read_rgb_image(path)

    if USE_DULL_RAZOR:
        image_rgb = dull_razor_remove_hair(image_rgb)

    if USE_MASK_CROP:
        image_rgb = crop_lesion_with_margin(
            image_rgb,
            mask_path,
            margin=CROP_MARGIN
        )
    elif USE_MASK_SEGMENTATION:
        image_rgb = apply_mask_segmentation(image_rgb, mask_path)

    return image_rgb.astype("float32")

def generate_multicrop_images_from_base(
    image_rgb,
    target_size,
    local_crop_ratio=0.80,
    include_full=True,
    use_horizontal_flip=True,
    use_vertical_flip=False,
    use_hv_flip=False
):
    image_rgb = image_rgb.astype("float32")
    h, w = image_rgb.shape[:2]

    crops = []

    if include_full:
        crops.append(cv2.resize(image_rgb, target_size))

    crop_h = int(h * local_crop_ratio)
    crop_w = int(w * local_crop_ratio)

    crop_h = max(1, min(crop_h, h))
    crop_w = max(1, min(crop_w, w))

    positions = [
        (0, 0),
        (0, w - crop_w),
        (h - crop_h, 0),
        (h - crop_h, w - crop_w),
        ((h - crop_h) // 2, (w - crop_w) // 2),
    ]

    for y, x in positions:
        y = max(0, int(y))
        x = max(0, int(x))

        crop = image_rgb[y:y + crop_h, x:x + crop_w]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, target_size)
        crops.append(crop)

    augmented = []

    for crop in crops:
        augmented.append(crop)

        if use_horizontal_flip:
            augmented.append(np.fliplr(crop))

        if use_vertical_flip:
            augmented.append(np.flipud(crop))

        if use_hv_flip:
            augmented.append(np.flipud(np.fliplr(crop)))

    return np.array(augmented, dtype="float32")

sample_row = test_df.sample(1, random_state=SEED).iloc[0]

base_img = preprocess_base_image(sample_row["image_path"], sample_row["mask_path"])

windows = generate_multicrop_images_from_base(
    base_img,
    target_size=IMG_SIZE_A,
    local_crop_ratio=LOCAL_CROP_RATIO,
    include_full=INCLUDE_FULL_IMAGE,
    use_horizontal_flip=USE_HORIZONTAL_FLIP,
    use_vertical_flip=USE_VERTICAL_FLIP,
    use_hv_flip=USE_HV_FLIP
)

print("Quantidade de imagens geradas:", len(windows))
print("Shape:", windows.shape)

n_show = min(len(windows), 12)

plt.figure(figsize=(14, 6))

for i in range(n_show):
    plt.subplot(2, 6, i + 1)
    plt.imshow(np.clip(windows[i], 0, 255).astype("uint8"))
    plt.title(f"Crop {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

from keras.layers import BatchNormalization

_original_batchnorm_init = BatchNormalization.__init__

def _patched_batchnorm_init(self, *args, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    _original_batchnorm_init(self, *args, **kwargs)

BatchNormalization.__init__ = _patched_batchnorm_init

model_a = keras.models.load_model(
    MODEL_A_PATH,
    compile=False,
    safe_mode=False
)

model_b = keras.models.load_model(
    MODEL_B_PATH,
    compile=False,
    safe_mode=False
)

model_c = keras.models.load_model(
    MODEL_C_PATH,
    compile=False,
    safe_mode=False
)

print("Modelo A input:", model_a.input_shape)
print("Modelo B input:", model_b.input_shape)
print("Modelo C input:", model_c.input_shape)

def predict_model_with_multicrop_tta(
    image_path,
    mask_path,
    img_size,
    model
):
    base_img = preprocess_base_image(image_path, mask_path)

    tta_images = generate_multicrop_images_from_base(
        base_img,
        target_size=img_size,
        local_crop_ratio=LOCAL_CROP_RATIO,
        include_full=INCLUDE_FULL_IMAGE,
        use_horizontal_flip=USE_HORIZONTAL_FLIP,
        use_vertical_flip=USE_VERTICAL_FLIP,
        use_hv_flip=USE_HV_FLIP
    )

    preds = model.predict(tta_images, verbose=0)

    return np.mean(preds, axis=0)


def predict_ensemble_multicrop_tta(image_path, mask_path):
    pred_a = predict_model_with_multicrop_tta(
        image_path=image_path,
        mask_path=mask_path,
        img_size=IMG_SIZE_A,
        model=model_a
    )

    pred_b = predict_model_with_multicrop_tta(
        image_path=image_path,
        mask_path=mask_path,
        img_size=IMG_SIZE_B,
        model=model_b
    )

    pred_c = predict_model_with_multicrop_tta(
        image_path=image_path,
        mask_path=mask_path,
        img_size=IMG_SIZE_C,
        model=model_c
    )

    final_pred = (
        WEIGHT_MODEL_A * pred_a +
        WEIGHT_MODEL_B * pred_b +
        WEIGHT_MODEL_C * pred_c
    )

    return final_pred

sample_row = test_df.iloc[0]

sample_pred = predict_ensemble_multicrop_tta(
    sample_row["image_path"],
    sample_row["mask_path"]
)

print("Shape da predição:", sample_pred.shape)
print("Soma:", sample_pred.sum())
print("Classe real:", sample_row["class_name"])
print("Classe predita:", class_name[int(np.argmax(sample_pred))])

start_time = time.time()

y_true_multicrop = []
y_pred_multicrop = []
y_prob_multicrop = []

for idx, row in test_df.reset_index(drop=True).iterrows():
    pred = predict_ensemble_multicrop_tta(row["image_path"], row["mask_path"])
    pred_class = int(np.argmax(pred))

    y_true_multicrop.append(int(row["label"]))
    y_pred_multicrop.append(pred_class)
    y_prob_multicrop.append(pred)

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Avaliadas {idx + 1}/{len(test_df)} imagens | tempo: {elapsed:.2f} s")

y_true_multicrop = np.array(y_true_multicrop)
y_pred_multicrop = np.array(y_pred_multicrop)
y_prob_multicrop = np.array(y_prob_multicrop)

elapsed_total = time.time() - start_time

print("Avaliação com Multi-Crop TTA concluída.")
print("Tempo total:", round(elapsed_total, 2), "segundos")
print("Tempo total:", round(elapsed_total / 60, 2), "minutos")

print("Relatório de Classificação do Ensemble DeepSMOTE com Multi-Crop TTA:")

classification_report_text = classification_report(
    y_true_multicrop,
    y_pred_multicrop,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(classification_report_text)

classification_report_dict = classification_report(
    y_true_multicrop,
    y_pred_multicrop,
    target_names=classes,
    digits=4,
    zero_division=0,
    output_dict=True
)

classification_report_df = pd.DataFrame(classification_report_dict).transpose()
classification_report_path = RESULTS_DIR / "classification_report_ensemble_oversampling.csv"
classification_report_df.to_csv(classification_report_path)

classification_report_txt_path = RESULTS_DIR / "classification_report_ensemble_oversampling.txt"
with open(classification_report_txt_path, "w", encoding="utf-8") as f:
    f.write(classification_report_text)

acc_multicrop = accuracy_score(y_true_multicrop, y_pred_multicrop)

macro_precision_multicrop = precision_score(y_true_multicrop, y_pred_multicrop, average="macro", zero_division=0)
macro_recall_multicrop = recall_score(y_true_multicrop, y_pred_multicrop, average="macro", zero_division=0)
macro_f1_multicrop = f1_score(y_true_multicrop, y_pred_multicrop, average="macro", zero_division=0)

weighted_precision_multicrop = precision_score(y_true_multicrop, y_pred_multicrop, average="weighted", zero_division=0)
weighted_recall_multicrop = recall_score(y_true_multicrop, y_pred_multicrop, average="weighted", zero_division=0)
weighted_f1_multicrop = f1_score(y_true_multicrop, y_pred_multicrop, average="weighted", zero_division=0)

print("\nMÉTRICAS GERAIS DO ENSEMBLE DEEPSMOTE COM MULTI-CROP TTA")
print(f"Accuracy: {acc_multicrop:.4f}")

print("\nMacro Average:")
print(f"  Precision: {macro_precision_multicrop:.4f}")
print(f"  Recall: {macro_recall_multicrop:.4f}")
print(f"  F1-Score: {macro_f1_multicrop:.4f}")

print("\nWeighted Average:")
print(f"  Precision: {weighted_precision_multicrop:.4f}")
print(f"  Recall: {weighted_recall_multicrop:.4f}")
print(f"  F1-Score: {weighted_f1_multicrop:.4f}")

cm_multicrop = confusion_matrix(
    y_true_multicrop,
    y_pred_multicrop,
    labels=list(range(len(classes)))
)

cm_percent_multicrop = cm_multicrop.astype("float") / cm_multicrop.sum(axis=1, keepdims=True) * 100
cm_percent_multicrop = np.nan_to_num(cm_percent_multicrop)

cm_abs_df = pd.DataFrame(
    cm_multicrop,
    index=classes,
    columns=classes
)

cm_abs_path = RESULTS_DIR / "matrix_confusion_ensemble_oversampling_absolute.csv"
cm_abs_df.to_csv(cm_abs_path)

print("\nMatriz de confusão absoluta:")
print(cm_abs_df)

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(cm_multicrop, cmap="Blues", vmin=0, vmax=200)

ax.set_title("Matrix Confusion - Model 14 (Ensemble with DeepSMOTE)", fontsize=16)
ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("True Class", fontsize=12)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

threshold = cm_multicrop.max() * 0.5

for i in range(cm_multicrop.shape[0]):
    for j in range(cm_multicrop.shape[1]):
        color = "white" if cm_multicrop[i, j] > threshold else "black"
        ax.text(
            j,
            i,
            str(cm_multicrop[i, j]),
            ha="center",
            va="center",
            color=color,
            fontsize=11
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Quantity")

plt.tight_layout()
cm_abs_img_path = RESULTS_DIR / "matrix_confusion_ensemble_oversampling_absolute.png"
plt.savefig(cm_abs_img_path, dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 7))
im = plt.imshow(
    cm_percent_multicrop,
    interpolation="nearest",
    cmap="Blues",
    vmin=0,
    vmax=100
)

plt.title("Matrix Confusion (%) - Model 5 (Ensemble with Oversampling)")
plt.colorbar(im, label="%")

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

for i in range(cm_percent_multicrop.shape[0]):
    for j in range(cm_percent_multicrop.shape[1]):
        color = "white" if cm_percent_multicrop[i, j] > 50 else "black"
        plt.text(
            j,
            i,
            f"{cm_percent_multicrop[i, j]:.1f}",
            ha="center",
            va="center",
            color=color,
            fontsize=8
        )

plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
cm_percent_img_path = RESULTS_DIR / "matrix_confusion_ensemble_oversampling_percent.png"
plt.savefig(cm_percent_img_path, dpi=300, bbox_inches="tight")
plt.show()

cm_percent_df = pd.DataFrame(
    cm_percent_multicrop,
    index=classes,
    columns=classes
)

print(cm_percent_df)

EXPERIMENT_NAME = (
    f"ensemble_deepsmote_multicrop_ratio{LOCAL_CROP_RATIO}_"
    f"A{WEIGHT_MODEL_A}_B{WEIGHT_MODEL_B}_C{WEIGHT_MODEL_C}"
).replace(".", "p")

predictions_df = test_df.reset_index(drop=True).copy()
predictions_df["y_true"] = y_true_multicrop
predictions_df["y_pred"] = y_pred_multicrop
predictions_df["class_true"] = [class_name[int(label)] for label in y_true_multicrop]
predictions_df["class_pred"] = [class_name[int(label)] for label in y_pred_multicrop]
predictions_df["correct"] = predictions_df["y_true"] == predictions_df["y_pred"]

for idx, class_label in enumerate(classes):
    predictions_df[f"prob_{class_label}"] = y_prob_multicrop[:, idx]

predictions_df["confidence_pred"] = [
    y_prob_multicrop[i, y_pred_multicrop[i]]
    for i in range(len(y_pred_multicrop))
]

predictions_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

EXAMPLES_DIR = RESULTS_DIR / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def make_example_grid(
    df,
    output_path,
    title,
    group_col,
    n_per_class=3,
    sort_col="confidence_pred"
):
    """
    Gera uma grade de exemplos por classe.

    - Verdadeiros positivos: use group_col="class_true".
    - Falsos positivos: use group_col="class_pred".
    """

    n_rows = len(classes)
    n_cols = n_per_class

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.6 * n_rows)
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    fig.suptitle(title, fontsize=16)

    for row_idx, class_label in enumerate(classes):
        subset = df[df[group_col] == class_label].copy()

        if sort_col in subset.columns:
            subset = subset.sort_values(sort_col, ascending=False)

        subset = subset.head(n_per_class)

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx >= len(subset):
                ax.text(
                    0.5,
                    0.5,
                    f"No example\n{class_label}",
                    ha="center",
                    va="center",
                    fontsize=10
                )
                continue

            row = subset.iloc[col_idx]

            try:
                image = preprocess_base_image(row["image_path"], row["mask_path"])
                image = np.clip(image, 0, 255).astype("uint8")
                ax.imshow(image)

                true_label = row["class_true"]
                pred_label = row["class_pred"]
                confidence = row["confidence_pred"]

                ax.set_title(
                    f"True: {true_label}\nPred: {pred_label}\nP={confidence:.3f}",
                    fontsize=10
                )

            except Exception as error:
                ax.text(
                    0.5,
                    0.5,
                    f"Error\n{row.get('image', '')}",
                    ha="center",
                    va="center",
                    fontsize=9
                )
                print(f"Erro ao carregar exemplo {row.get('image', '')}: {error}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


true_positive_examples = predictions_df[predictions_df["correct"] == True].copy()
false_positive_examples = predictions_df[predictions_df["correct"] == False].copy()

tp_examples_path = RESULTS_DIR / "examples_true_positives.csv"
fp_examples_path = RESULTS_DIR / "examples_false_positives.csv"

true_positive_examples.to_csv(tp_examples_path, index=False)
false_positive_examples.to_csv(fp_examples_path, index=False)

tp_img_path = EXAMPLES_DIR / "examples_true_positives_by_class.png"
fp_img_path = EXAMPLES_DIR / "examples_false_positives_by_predicted_class.png"

make_example_grid(
    df=true_positive_examples,
    output_path=tp_img_path,
    title="True Positive Examples - Model 5 Ensemble with Oversampling",
    group_col="class_true",
    n_per_class=3,
    sort_col="confidence_pred"
)

make_example_grid(
    df=false_positive_examples,
    output_path=fp_img_path,
    title="False Positive Examples by Predicted Class - Model 5 Ensemble with Oversampling",
    group_col="class_pred",
    n_per_class=3,
    sort_col="confidence_pred"
)

metrics = {
    "experiment_name": EXPERIMENT_NAME,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "configuration": {
        "model_a": {
            "name": "DenseNet201 + DeepSMOTE",
            "img_size": IMG_SIZE_A,
            "path": str(MODEL_A_PATH),
            "weight": WEIGHT_MODEL_A
        },
        "model_b": {
            "name": "DenseNet169 + DeepSMOTE",
            "img_size": IMG_SIZE_B,
            "path": str(MODEL_B_PATH),
            "weight": WEIGHT_MODEL_B
        },
        "model_c": {
            "name": "EfficientNetB4 + DeepSMOTE",
            "img_size": IMG_SIZE_C,
            "path": str(MODEL_C_PATH),
            "weight": WEIGHT_MODEL_C
        },
        "use_dull_razor": USE_DULL_RAZOR,
        "use_mask_segmentation": USE_MASK_SEGMENTATION,
        "use_mask_crop": USE_MASK_CROP,
        "crop_margin": CROP_MARGIN,
        "multicrop_tta": True,
        "local_crop_ratio": LOCAL_CROP_RATIO,
        "include_full_image": INCLUDE_FULL_IMAGE,
        "use_horizontal_flip": USE_HORIZONTAL_FLIP,
        "use_vertical_flip": USE_VERTICAL_FLIP,
        "use_hv_flip": USE_HV_FLIP,
        "confusion_matrix_absolute_plot": {
            "title": "Matrix Confusion - Model 5 (Ensemble with Oversampling)",
            "cmap": "Blues",
            "vmin": 0,
            "vmax": 200
        }
    },
    "metrics": {
        "accuracy": float(acc_multicrop),
        "macro_precision": float(macro_precision_multicrop),
        "macro_recall": float(macro_recall_multicrop),
        "macro_f1": float(macro_f1_multicrop),
        "weighted_precision": float(weighted_precision_multicrop),
        "weighted_recall": float(weighted_recall_multicrop),
        "weighted_f1": float(weighted_f1_multicrop),
        "elapsed_seconds": float(elapsed_total),
        "elapsed_minutes": float(elapsed_total / 60)
    }
}

metrics_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

cm_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_confusion_matrix_percent.csv"
cm_percent_df.to_csv(cm_path)

ensemble_reference_path = RESULTS_DIR / "ensemble_model5_reference.json"
ensemble_reference = {
    "model_a": {"name": "DenseNet201 + DeepSMOTE", "path": str(MODEL_A_PATH), "img_size": IMG_SIZE_A, "weight": WEIGHT_MODEL_A},
    "model_b": {"name": "DenseNet169 + DeepSMOTE", "path": str(MODEL_B_PATH), "img_size": IMG_SIZE_B, "weight": WEIGHT_MODEL_B},
    "model_c": {"name": "EfficientNetB4 + DeepSMOTE", "path": str(MODEL_C_PATH), "img_size": IMG_SIZE_C, "weight": WEIGHT_MODEL_C},
    "preprocessing": {
        "use_dull_razor": USE_DULL_RAZOR,
        "use_mask_segmentation": USE_MASK_SEGMENTATION,
        "use_mask_crop": USE_MASK_CROP,
        "crop_margin": CROP_MARGIN
    },
    "multicrop_tta": {
        "local_crop_ratio": LOCAL_CROP_RATIO,
        "include_full_image": INCLUDE_FULL_IMAGE,
        "use_horizontal_flip": USE_HORIZONTAL_FLIP,
        "use_vertical_flip": USE_VERTICAL_FLIP,
        "use_hv_flip": USE_HV_FLIP
    }
}
with open(ensemble_reference_path, "w", encoding="utf-8") as f:
    json.dump(ensemble_reference, f, indent=4, ensure_ascii=False)

print("Resultados salvos com sucesso:")
print("Predições:", predictions_path)
print("Relatório CSV:", classification_report_path)
print("Relatório TXT:", classification_report_txt_path)
print("Métricas:", metrics_path)
print("Referência do ensemble:", ensemble_reference_path)
print("Matriz absoluta CSV:", cm_abs_path)
print("Matriz absoluta PNG:", cm_abs_img_path)
print("Matriz percentual CSV:", cm_path)
print("Matriz percentual PNG:", cm_percent_img_path)
print("Exemplos TP CSV:", tp_examples_path)
print("Exemplos FP CSV:", fp_examples_path)
print("Exemplos TP PNG:", tp_img_path)
print("Exemplos FP PNG:", fp_img_path)