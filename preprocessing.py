import numpy as np
import pandas as pd
import os
import tensorflow as tf
import seaborn as sns
import keras

from matplotlib import pyplot as plt
from keras import layers
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

os.environ["KERAS_BACKEND"] = "tensorflow"

BATCH_SIZE = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
CSV_PATH = os.path.join(DATA_DIR, "GroundTruth.csv")

metadata = pd.read_csv(CSV_PATH)
print(metadata.head())
print(metadata.info())

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

class_code = {label: code for code, label in enumerate(classes)}
print(class_code)

class_name = {code: label for code, label in enumerate(classes)}
print(class_name)

def get_coded_labels(directory, metadata, class_code):
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    image_names = [x[:-4] for x in os.listdir(directory) if x.lower().endswith((".jpg", ".jpeg", ".png"))]

    image_df = pd.DataFrame({"image": sorted(image_names)})

    merged = image_df.merge(metadata, on="image", how="inner")

    coded_labels = merged[classes].idxmax(axis=1).map(lambda x: class_code[x])

    return list(coded_labels)

dataset = keras.utils.image_dataset_from_directory(
    directory=IMAGES_DIR,
    labels=get_coded_labels(IMAGES_DIR, metadata, class_code),
    batch_size=None,  # sem batch aqui
    label_mode="categorical",
    shuffle=True,
    image_size=(128, 128)
)

train_ds, test_ds = keras.utils.split_dataset(
    dataset,
    left_size=0.8,
    shuffle=True,
    seed=42
)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

tf.data.DatasetSpec.from_value(train_ds)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(5, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        label_index = np.argmax(labels[i].numpy())
        plt.title(class_name[label_index])
        plt.axis("off")

plt.show()

train_label_df = pd.DataFrame(columns=classes)

for images, labels in train_ds.as_numpy_iterator():
    batch_df = pd.DataFrame(labels, columns=classes)
    train_label_df = pd.concat([train_label_df, batch_df], ignore_index=True)

card = len(train_label_df)
print("Quantidade de imagens no conjunto de treino:", card)

label_counts = train_label_df.sum()

plt.figure(figsize=(10, 5))
ax = sns.barplot(x=label_counts.index, y=label_counts.values)

for container in ax.containers:
    ax.bar_label(container)

plt.title("Distribuição das classes no conjunto de treino")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

print(label_counts)

inputs = keras.Input(shape=(128, 128, 3))

base_model = InceptionV3(weights="imagenet", include_top=False)
base_model.trainable = False

x = layers.Rescaling(scale=1.0 / 255)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_code.keys()), activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

history = model.fit(
    train_ds,
    epochs=5,
    callbacks=callbacks,
    validation_data=test_ds,
)

base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

history2 = model.fit(
    train_ds,
    epochs=1,
    callbacks=callbacks,
    validation_data=test_ds,
)

plt.figure(figsize=(10, 10))
test_ds_vis = test_ds.shuffle(100)

for images, labels in test_ds_vis.take(1):
    predictions = model.predict(images)

    for i in range(min(5, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))

        true_label = class_name[np.argmax(labels[i].numpy())]
        pred_label = class_name[np.argmax(predictions[i])]

        plt.title(f"{true_label} → {pred_label}")
        plt.axis("off")

plt.show()