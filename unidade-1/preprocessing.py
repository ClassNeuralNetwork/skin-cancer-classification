import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import keras

from matplotlib import pyplot as plt
from keras import layers
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report

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

    image_names = [
        x[:-4]
        for x in os.listdir(directory)
        if x.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_df = pd.DataFrame({"image": sorted(image_names)})

    merged = image_df.merge(metadata, on="image", how="inner")

    coded_labels = merged[classes].idxmax(axis=1).map(lambda x: class_code[x])

    return list(coded_labels)

dataset = keras.utils.image_dataset_from_directory(
    directory=IMAGES_DIR,
    labels=get_coded_labels(IMAGES_DIR, metadata, class_code),
    batch_size=None,
    label_mode="categorical",
    shuffle=True,
    image_size=(256, 256)
)

train_ds, test_ds = keras.utils.split_dataset(
    dataset,
    left_size=0.8,
    shuffle=True,
    seed=42
)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images, labels):
    for layer in data_augmentation_layers:
        images = layer(images, training=True)
    return images, labels

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

tf.data.DatasetSpec.from_value(train_ds)

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(min(5, len(images))):
        ax = plt.subplot(3, 3, i + 1)

        image = tf.cast(tf.clip_by_value(images[i], 0, 255), tf.uint8)
        plt.imshow(image)

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

inputs = keras.Input(shape=(256, 256, 3))

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)

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
    epochs=5,
    callbacks=callbacks,
    validation_data=test_ds,
)

plt.figure(figsize=(10, 10))
test_ds_vis = test_ds.shuffle(100)

for images, labels in test_ds_vis.take(1):
    predictions = model.predict(images)

    for i in range(min(5, len(images))):
        ax = plt.subplot(3, 3, i + 1)

        image = tf.cast(tf.clip_by_value(images[i], 0, 255), tf.uint8)
        plt.imshow(image)

        true_label = class_name[np.argmax(labels[i].numpy())]
        pred_label = class_name[np.argmax(predictions[i])]

        plt.title(f"{true_label} → {pred_label}")
        plt.axis("off")

plt.show()

full_history = {}

for key in history.history.keys():
    full_history[key] = history.history[key] + history2.history.get(key, [])

plt.figure(figsize=(8, 5))
plt.plot(full_history["loss"], label="train_loss")
plt.plot(full_history["val_loss"], label="val_loss")
plt.title("Loss por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(full_history["accuracy"], label="train_accuracy")
plt.plot(full_history["val_accuracy"], label="val_accuracy")
plt.title("Accuracy por época")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(full_history["recall"], label="train_recall")
plt.plot(full_history["val_recall"], label="val_recall")
plt.title("Recall por época")
plt.xlabel("Época")
plt.ylabel("Recall")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(full_history["precision"], label="train_precision")
plt.plot(full_history["val_precision"], label="val_precision")
plt.title("Precision por época")
plt.xlabel("Época")
plt.ylabel("Precision")
plt.legend()
plt.show()

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues"
)
plt.title("Matriz de Confusão")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.show()

report = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    digits=4
)

print("Relatório de Classificação:\n")
print(report)

final_train_acc = full_history["accuracy"][-1]
final_val_acc = full_history["val_accuracy"][-1]
final_train_loss = full_history["loss"][-1]
final_val_loss = full_history["val_loss"][-1]

print("\nDiagnóstico final:")
print(f"Train accuracy: {final_train_acc:.4f}")
print(f"Val accuracy: {final_val_acc:.4f}")
print(f"Train loss: {final_train_loss:.4f}")
print(f"Val loss: {final_val_loss:.4f}")

if final_train_acc - final_val_acc > 0.10 and final_val_loss > final_train_loss:
    print("Possível overfitting.")
elif final_train_acc < 0.70 and final_val_acc < 0.70:
    print("Possível underfitting ou modelo ainda pouco ajustado.")
else:
    print("Sem sinal forte de overfitting clássico; avaliar matriz de confusão e métricas por classe.")