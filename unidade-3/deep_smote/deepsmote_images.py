# %% [markdown]
# # 01 - Gerar Dataset DeepSMOTE com Crop pela Máscara
# 
# Pipeline usado:
# 
# 1) imagem original
# 2) crop pela máscara
# 3) resize para o tamanho do DeepSMOTE
# 4) treina autoencoder
# 5) gera amostras sintéticas no espaço latente
# 6) decodifica imagens sintéticas
# 7) salva imagens sintéticas no computador
# 8) salva CSV com metadados

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
    from keras import layers, models

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors

    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((1, _cell_elapsed_time))
    print("\nTempo da célula 1: {:.2f} segundos".format(_cell_elapsed_time))

# %%
tf.config.list_physical_devices('GPU')

# %% [markdown]
# 

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

    IMAGES_DIR = DATA_DIR / "images"
    MASKS_DIR = DATA_DIR / "masks"
    CSV_PATH = DATA_DIR / "GroundTruth.csv"

    OUTPUT_DIR = DATA_DIR / "deepsmote"
    SYNTH_IMAGES_DIR = OUTPUT_DIR / "images"
    SYNTH_CSV_PATH = OUTPUT_DIR / "metadata_deepsmote.csv"

    MODELS_DIR = Path("../../models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    AUTOENCODER_PATH = MODELS_DIR / "deepsmote_autoencoder_crop_160_latent512.keras"
    ENCODER_PATH = MODELS_DIR / "deepsmote_encoder_crop_160_latent512.keras"
    DECODER_PATH = MODELS_DIR / "deepsmote_decoder_crop_160_latent512.keras"

    TEST_SIZE = 0.20

    DEEPSMOTE_IMG_SIZE = (256, 256)

    BATCH_SIZE = 16
    AE_EPOCHS = 120
    LATENT_DIM = 512
    LEARNING_RATE = 1e-4

    USE_MASK_CROP = True
    CROP_MARGIN = 20

    TRAIN_AUTOENCODER = True

    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    class_code = {label: idx for idx, label in enumerate(classes)}
    class_name = {idx: label for label, idx in class_code.items()}

    print("DATA_DIR:", DATA_DIR.resolve())
    print("DEEPSMOTE_IMG_SIZE:", DEEPSMOTE_IMG_SIZE)
    print("LATENT_DIM:", LATENT_DIM)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("AE_EPOCHS:", AE_EPOCHS)
    print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((2, _cell_elapsed_time))
    print("\nTempo da célula 2: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 2. Funções de leitura, máscara e crop

# %%
import time as _time
_cell_start_time = _time.time()

try:
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
        if matches:
            return str(matches[0])

        return None


    def read_rgb_image(path):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Não foi possível carregar a imagem: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


    def read_binary_mask(mask_path, target_shape):
        if mask_path is None or pd.isna(mask_path):
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        height, width = target_shape[:2]
        mask = cv2.resize(mask, (width, height))
        binary_mask = (mask > 127).astype(np.uint8)

        if binary_mask.sum() == 0:
            return None

        return binary_mask


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


    def preprocess_for_deepsmote(image_path, mask_path=None, target_size=DEEPSMOTE_IMG_SIZE):
        image_rgb = read_rgb_image(image_path)

        if USE_MASK_CROP:
            image_rgb = crop_lesion_with_margin(
                image_rgb,
                mask_path,
                margin=CROP_MARGIN
            )

        image_rgb = cv2.resize(image_rgb, target_size)
        image_rgb = image_rgb.astype("float32") / 127.5 - 1.0  

        return image_rgb


    def to_display(img):
        img = np.array(img).astype("float32")
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        return np.clip(img, 0.0, 1.0)

    print("Funções definidas.")
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((3, _cell_elapsed_time))
    print("\nTempo da célula 3: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 3. Carregar metadados e dividir treino/teste

# %%
import time as _time
_cell_start_time = _time.time()

try:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em: {CSV_PATH.resolve()}\n"
            "Se estiver executando na raiz do projeto, troque DATA_DIR para Path('data')."
        )

    metadata = pd.read_csv(CSV_PATH)

    rows = []
    for _, row in metadata.iterrows():
        image_id = str(row["image"])
        image_path = find_image_path(image_id, IMAGES_DIR)

        if image_path is None:
            continue

        label_name = row[classes].idxmax()
        label = class_code[label_name]
        mask_path = find_mask_path(image_id, MASKS_DIR)

        rows.append({
            "image": image_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "class_name": label_name,
            "label": label,
            "is_synthetic": 0
        })

    df = pd.DataFrame(rows)

    print("Total de imagens localizadas:", len(df))
    print("Máscaras encontradas:", df["mask_path"].notna().sum())
    print("\nDistribuição original:")
    print(df["class_name"].value_counts().reindex(classes))

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train_real_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_real_split.csv", index=False)

    print("\nTreino:", len(train_df))
    print("Teste:", len(test_df))
    print("\nDistribuição treino:")
    print(train_df["class_name"].value_counts().reindex(classes))
    print("\nDistribuição teste:")
    print(test_df["class_name"].value_counts().reindex(classes))
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((4, _cell_elapsed_time))
    print("\nTempo da célula 4: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 4. Criar dataset do autoencoder em batches

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def deepsmote_generator(dataframe):
        paths = dataframe["image_path"].values
        masks = dataframe["mask_path"].values

        for image_path, mask_path in zip(paths, masks):
            img = preprocess_for_deepsmote(
                image_path=image_path,
                mask_path=mask_path,
                target_size=DEEPSMOTE_IMG_SIZE
            )
            yield img, img


    output_signature = (
        tf.TensorSpec(shape=(*DEEPSMOTE_IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*DEEPSMOTE_IMG_SIZE, 3), dtype=tf.float32),
    )

    ae_ds = tf.data.Dataset.from_generator(
        lambda: deepsmote_generator(train_df),
        output_signature=output_signature
    )

    ae_ds = (
        ae_ds
        .shuffle(1024, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Dataset do autoencoder criado.")
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((5, _cell_elapsed_time))
    print("\nTempo da célula 5: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 5. Construir autoencoder

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def build_autoencoder(input_shape, latent_dim):
        encoder_inputs = layers.Input(shape=input_shape, name="encoder_input")

        x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        shape_before_flatten = tuple(int(dim) for dim in x.shape[1:])
        x = layers.Flatten()(x)
        latent = layers.Dense(latent_dim, name="latent_vector")(x)

        encoder = models.Model(
            encoder_inputs,
            latent,
            name="deepsmote_encoder"
        )

        decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
        flatten_units = int(np.prod(shape_before_flatten))

        x = layers.Dense(flatten_units, activation="relu")(decoder_inputs)
        x = layers.Reshape(shape_before_flatten)(x)

        x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        decoder_outputs = layers.Conv2D(
            3,
            3,
            padding="same",
            activation="tanh",
            name="decoder_output"
        )(x)

        decoder = models.Model(
            decoder_inputs,
            decoder_outputs,
            name="deepsmote_decoder"
        )

        autoencoder_inputs = encoder_inputs
        autoencoder_outputs = decoder(encoder(autoencoder_inputs))

        autoencoder = models.Model(
            autoencoder_inputs,
            autoencoder_outputs,
            name="deepsmote_autoencoder"
        )

        return autoencoder, encoder, decoder


    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=(*DEEPSMOTE_IMG_SIZE, 3),
        latent_dim=LATENT_DIM
    )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mae"
    )

    autoencoder.summary()
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((6, _cell_elapsed_time))
    print("\nTempo da célula 6: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 6. Treinar ou carregar autoencoder

# %%
import time as _time
_cell_start_time = _time.time()

try:
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            AUTOENCODER_PATH,
            monitor="loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        )
    ]

    if TRAIN_AUTOENCODER or not AUTOENCODER_PATH.exists():
        history = autoencoder.fit(
            ae_ds,
            epochs=AE_EPOCHS,
            callbacks=callbacks
        )

        autoencoder.save(AUTOENCODER_PATH)
        encoder.save(ENCODER_PATH)
        decoder.save(DECODER_PATH)

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="loss")
        plt.title("Treinamento do Autoencoder DeepSMOTE")
        plt.xlabel("Época")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "curvas_treinamento_autoencoder.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        autoencoder = keras.models.load_model(AUTOENCODER_PATH)
        encoder = keras.models.load_model(ENCODER_PATH)
        decoder = keras.models.load_model(DECODER_PATH)

    print("Autoencoder pronto.")
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((7, _cell_elapsed_time))
    print("\nTempo da célula 7: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 7. Visualizar reconstruções

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def load_preview_images(dataframe, n=8):
        sample_df = dataframe.sample(n=min(n, len(dataframe)), random_state=SEED)
        imgs = []

        for _, row in sample_df.iterrows():
            imgs.append(
                preprocess_for_deepsmote(
                    row["image_path"],
                    row["mask_path"],
                    target_size=DEEPSMOTE_IMG_SIZE
                )
            )

        return np.array(imgs, dtype="float32")


    preview = load_preview_images(train_df, n=8)
    recon = autoencoder.predict(preview, batch_size=4, verbose=0)

    fig, axes = plt.subplots(2, len(preview), figsize=(3 * len(preview), 6))

    for i in range(len(preview)):
        axes[0, i].imshow(to_display(preview[i]))
        axes[0, i].set_title("Real")
        axes[0, i].axis("off")

        axes[1, i].imshow(to_display(recon[i]))
        axes[1, i].set_title("Recon.")
        axes[1, i].axis("off")

    plt.suptitle("Reconstrução pelo Autoencoder")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reconstrucao_autoencoder.png", dpi=300, bbox_inches="tight")
    plt.show()

    del preview, recon
    gc.collect()
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((8, _cell_elapsed_time))
    print("\nTempo da célula 8: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 8. Codificar imagens reais do treino em batches

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def encode_dataframe_in_batches(dataframe, batch_size=16):
        z_list = []
        y_list = []

        total = len(dataframe)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_df = dataframe.iloc[start:end]

            imgs = []
            labels = []

            for _, row in batch_df.iterrows():
                img = preprocess_for_deepsmote(
                    row["image_path"],
                    row["mask_path"],
                    target_size=DEEPSMOTE_IMG_SIZE
                )
                imgs.append(img)
                labels.append(int(row["label"]))

            imgs = np.array(imgs, dtype="float32")
            z = encoder.predict(imgs, batch_size=batch_size, verbose=0)

            z_list.append(z.astype("float32"))
            y_list.append(np.array(labels, dtype="int32"))

            del imgs, z
            gc.collect()

            print(f"Codificadas {end}/{total} imagens")

        Z = np.concatenate(z_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        return Z, y


    Z_train, y_train = encode_dataframe_in_batches(
        train_df,
        batch_size=BATCH_SIZE
    )

    np.save(OUTPUT_DIR / "Z_train_real.npy", Z_train)
    np.save(OUTPUT_DIR / "y_train_real.npy", y_train)

    print("Z_train:", Z_train.shape)
    print("y_train:", y_train.shape)
    print("Distribuição real:")
    print(Counter(y_train))
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((9, _cell_elapsed_time))
    print("\nTempo da célula 9: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 9. Gerar vetores latentes sintéticos por classe

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def generate_latent_deepsmote(Z, y, target_count=None, k_neighbors=5):
        synthetic_z = []
        synthetic_y = []

        counts = Counter(y)

        if target_count is None:
            target_count = max(counts.values())

        print("Contagem original:", counts)
        print("Target por classe:", target_count)

        rng = np.random.default_rng(SEED)

        for label in sorted(counts.keys()):
            class_indices = np.where(y == label)[0]
            Z_class = Z[class_indices]

            n_current = len(Z_class)
            n_to_generate = target_count - n_current

            print(f"{class_name[int(label)]}: atual={n_current}, gerar={max(n_to_generate, 0)}")

            if n_to_generate <= 0:
                continue

            if n_current < 2:
                print(f"Classe {label} ignorada: poucas amostras.")
                continue

            n_neighbors = min(k_neighbors + 1, n_current)

            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(Z_class)

            _, indices = nn.kneighbors(Z_class)

            for _ in range(n_to_generate):
                base_pos = rng.integers(0, n_current)

                if n_neighbors > 1:
                    neighbor_position = rng.integers(1, n_neighbors)
                else:
                    neighbor_position = 0

                neighbor_pos = indices[base_pos, neighbor_position]

                z_base = Z_class[base_pos]
                z_neighbor = Z_class[neighbor_pos]

                alpha = rng.uniform(0.2, 0.8)
                z_new = z_base + alpha * (z_neighbor - z_base)

                synthetic_z.append(z_new.astype("float32"))
                synthetic_y.append(int(label))

        if not synthetic_z:
            return np.empty((0, Z.shape[1]), dtype="float32"), np.empty((0,), dtype="int32")

        return np.array(synthetic_z, dtype="float32"), np.array(synthetic_y, dtype="int32")


    Z_synth, y_synth = generate_latent_deepsmote(
        Z_train,
        y_train,
        target_count=None,
        k_neighbors=5
    )

    np.save(OUTPUT_DIR / "Z_synthetic.npy", Z_synth)
    np.save(OUTPUT_DIR / "y_synthetic.npy", y_synth)

    print("Z_synth:", Z_synth.shape)
    print("y_synth:", y_synth.shape)
    print("Distribuição sintética:")
    print(Counter(y_synth))
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((10, _cell_elapsed_time))
    print("\nTempo da célula 10: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 10. Decodificar e salvar imagens sintéticas em disco

# %%
import time as _time
_cell_start_time = _time.time()

try:
    def save_synthetic_images_in_batches(decoder, Z_synthetic, y_synthetic, batch_size=16):
        SYNTH_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        synth_rows = []
        total = len(Z_synthetic)
        global_index = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            Z_batch = Z_synthetic[start:end]
            y_batch = y_synthetic[start:end]

            X_batch = decoder.predict(
                Z_batch,
                batch_size=batch_size,
                verbose=0
            )
        
            X_batch = (X_batch + 1.0) / 2.0
            X_batch = np.clip(X_batch, 0.0, 1.0)

            for img, label in zip(X_batch, y_batch):
                label = int(label)
                label_name = class_name[label]

                class_dir = SYNTH_IMAGES_DIR / label_name
                class_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{label_name}_synth_{global_index:06d}.jpg"
                save_path = class_dir / filename

                img_uint8 = (img * 255).astype("uint8")
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

                cv2.imwrite(str(save_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                synth_rows.append({
                    "image": filename.replace(".jpg", ""),
                    "image_path": str(save_path),
                    "mask_path": None,
                    "class_name": label_name,
                    "label": label,
                    "is_synthetic": 1
                })

                global_index += 1

            del X_batch
            gc.collect()

            print(f"Salvas {end}/{total} imagens sintéticas")

        synth_df = pd.DataFrame(synth_rows)
        synth_df.to_csv(SYNTH_CSV_PATH, index=False)

        print("CSV sintético salvo em:", SYNTH_CSV_PATH)
        print("Total sintéticas salvas:", len(synth_df))

        return synth_df


    synth_df = save_synthetic_images_in_batches(
        decoder=decoder,
        Z_synthetic=Z_synth,
        y_synthetic=y_synth,
        batch_size=BATCH_SIZE
    )

    print(synth_df["class_name"].value_counts().reindex(classes))
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((11, _cell_elapsed_time))
    print("\nTempo da célula 11: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## 11. Visualizar imagens sintéticas salvas

# %%
import time as _time
_cell_start_time = _time.time()

try:
    sample_synth = synth_df.sample(n=min(12, len(synth_df)), random_state=SEED)

    cols = 6
    rows = int(np.ceil(len(sample_synth) / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))

    for i, (_, row) in enumerate(sample_synth.iterrows()):
        img = read_rgb_image(row["image_path"])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(row["class_name"])
        plt.axis("off")

    plt.suptitle("Exemplos de imagens sintéticas DeepSMOTE salvas")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exemplos_imagens_sinteticas_deepsmote.png", dpi=300, bbox_inches="tight")
    plt.show()
finally:
    _cell_elapsed_time = _time.time() - _cell_start_time
    _notebook_total_time = globals().get('_notebook_total_time', 0.0) + _cell_elapsed_time
    _notebook_cell_times = globals().get('_notebook_cell_times', [])
    _notebook_cell_times.append((12, _cell_elapsed_time))
    print("\nTempo da célula 12: {:.2f} segundos".format(_cell_elapsed_time))

# %% [markdown]
# ## Tempo total de execução

# %%
total_seconds = globals().get('_notebook_total_time', 0.0)
total_minutes = total_seconds / 60

print("Tempo total de execução: {:.2f} segundos".format(total_seconds))
print("Tempo total de execução: {:.2f} minutos".format(total_minutes))


