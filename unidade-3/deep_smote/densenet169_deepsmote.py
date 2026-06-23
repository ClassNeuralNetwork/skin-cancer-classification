# %% [markdown]
# # 02 — Treinar DenseNet169 com Dataset DeepSMOTE Salvo

# %% [markdown]
# ## Controle de tempo
# 
# Cada célula de código imprime o tempo gasto em **segundos**. Ao final do notebook, a última célula soma o tempo total e mostra o resultado em segundos e minutos.

# %%
import time as _time
_notebook_total_time = 0.0
_notebook_cell_times = []
_cell_start_time = _time.time()

try:
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import gc
    import random
    from pathlib import Path
    from collections import Counter

    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import tensorflow as tf
    import keras
    from keras import layers, models, regularizers

    from tensorflow.keras.applications import DenseNet169
    from tensorflow.keras.applications.densenet import preprocess_input

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((1, _cell_elapsed_time))
    print("\nTempo da célula 1: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 1. Configurações

# %%
import time as _time
_cell_start_time = _time.time()

try:
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    DATA_DIR = Path("../../data")
    DEEPSMOTE_DIR = DATA_DIR / "deepsmote"

    TRAIN_REAL_CSV = DEEPSMOTE_DIR / "train_real_split.csv"
    TEST_REAL_CSV = DEEPSMOTE_DIR / "test_real_split.csv"
    SYNTH_CSV = DEEPSMOTE_DIR / "metadata_deepsmote.csv"

    MODELS_DIR = Path("../../models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR = Path("../../results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    IMG_SIZE = (320, 320)
    BATCH_SIZE = 16
    EPOCHS = 120
    LEARNING_RATE = 1e-4

    BEST_MODEL_PATH = MODELS_DIR / "best_densenet169_deepsmote_saved_images.keras"
    FINAL_MODEL_PATH = MODELS_DIR / "densenet169_deepsmote_saved_images_final.keras"

    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    class_code = {label: idx for idx, label in enumerate(classes)}
    class_name = {idx: label for label, idx in class_code.items()}

    print("IMG_SIZE:", IMG_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("BEST_MODEL_PATH:", BEST_MODEL_PATH)
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((2, _cell_elapsed_time))
    print("\nTempo da célula 2: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 2. Carregar CSVs salvos

# %%
import time as _time
_cell_start_time = _time.time()

try:
    for path in [TRAIN_REAL_CSV, TEST_REAL_CSV, SYNTH_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path.resolve()}\n"
                "Execute primeiro o notebook 01_generate_deepsmote_dataset.ipynb."
            )

    train_real_df = pd.read_csv(TRAIN_REAL_CSV)
    test_df = pd.read_csv(TEST_REAL_CSV)
    synth_df = pd.read_csv(SYNTH_CSV)

    train_real_df["is_synthetic"] = 0
    synth_df["is_synthetic"] = 1

    train_df = pd.concat(
        [train_real_df, synth_df],
        ignore_index=True
    )

    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print("Treino real:", len(train_real_df))
    print("Sintéticas:", len(synth_df))
    print("Treino total:", len(train_df))
    print("Teste:", len(test_df))

    print("\nDistribuição treino total:")
    print(train_df["class_name"].value_counts().reindex(classes))

    print("\nDistribuição teste:")
    print(test_df["class_name"].value_counts().reindex(classes))
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((3, _cell_elapsed_time))
    print("\nTempo da célula 3: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 3. Criar pipeline tf.data

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def load_image_tf(image_path, label):
        image = tf.io.read_file(image_path)

        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)

        return image, label


    def make_dataset(dataframe, shuffle=False):
        image_paths = dataframe["image_path"].astype(str).values
        labels = dataframe["label"].astype("int32").values

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(load_image_tf, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(dataframe), 4096), seed=SEED)

        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return ds


    train_ds = make_dataset(train_df, shuffle=True)
    test_ds = make_dataset(test_df, shuffle=False)

    print("Datasets criados.")
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((4, _cell_elapsed_time))
    print("\nTempo da célula 4: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 4. DenseNet169

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def build_densenet169_classifier(input_shape=(320, 320, 3), n_classes=7):
        inputs = layers.Input(shape=input_shape)

        x = preprocess_input(inputs)

        base_model = DenseNet169(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

        base_model.trainable = False

        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(n_classes, activation="softmax")(x)

        model = models.Model(
            inputs=inputs,
            outputs=outputs,
            name="densenet169_deepsmote_saved_images"
        )

        return model


    model = build_densenet169_classifier(
        input_shape=(*IMG_SIZE, 3),
        n_classes=len(classes)
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((5, _cell_elapsed_time))
    print("\nTempo da célula 5: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 5. Treinamento

# %%
import time as _time
_cell_start_time = _time.time()

try:
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=18,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(FINAL_MODEL_PATH)
    print("Modelo final salvo em:", FINAL_MODEL_PATH)
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((6, _cell_elapsed_time))
    print("\nTempo da célula 6: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 6. Curvas de treinamento

# %%
import time as _time
_cell_start_time = _time.time()

try:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação/Teste")
    plt.title("Acurácia - DenseNet169 + DeepSMOTE salvo")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "curva_accuracy_densenet169_deepsmote_salvo.png", dpi=300, bbox_inches="tight")
    plt.imshow()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação/Teste")
    plt.title("Loss - DenseNet169 + DeepSMOTE salvo")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "curva_loss_densenet169_deepsmote_salvo.png", dpi=300, bbox_inches="tight")
    plt.imshow()
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((7, _cell_elapsed_time))
    print("\nTempo da célula 7: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 7. Avaliação final

# %%
import time as _time
_cell_start_time = _time.time()

try:
    best_model = keras.models.load_model(BEST_MODEL_PATH)

    y_true = test_df["label"].astype("int32").values
    y_prob = best_model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("Accuracy:", acc)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)
    print("Weighted F1:", weighted_f1)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0
    ))

    metrics_df = pd.DataFrame([{
        "model": "DenseNet169 + DeepSMOTE saved images",
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }])

    metrics_df.to_csv(RESULTS_DIR / "metricas_densenet169_deepsmote_salvo.csv", index=False)
    metrics_df
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((8, _cell_elapsed_time))
    print("\nTempo da célula 8: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 8. Matriz de confusão percentual

# %%
import time as _time
_cell_start_time = _time.time()

try:
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_percent, interpolation="nearest")
    plt.title("Matriz de Confusão (%) - DenseNet169 + DeepSMOTE salvo")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}",
                ha="center",
                va="center"
            )

    plt.ylabel("Classe real")
    plt.xlabel("Classe predita")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "matriz_confusao_densenet169_deepsmote_salvo.png", dpi=300, bbox_inches="tight")
    plt.imsave()
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((9, _cell_elapsed_time))
    print("\nTempo da célula 9: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## Tempo total de execução

# %%
total_seconds = globals().get('_notebook_total_time', 0.0)
total_minutes = total_seconds / 60

print("Tempo total de execução: {:.2f} segundos".format(total_seconds))
print("Tempo total de execução: {:.2f} minutos".format(total_minutes))


