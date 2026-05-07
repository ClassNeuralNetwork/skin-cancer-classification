import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "GroundTruth.csv")

IMG_SIZE = (64, 64)
NUM_CLASSES = 7
classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

metadata = pd.read_csv(CSV_PATH)

print(metadata.head())
metadata.info()

X = []
y = []

for _, row in metadata.iterrows():
  image_name = row["image"] + ".jpg"
  image_path = os.path.join(IMAGES_DIR, image_name)

  if os.path.exists(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = img / 255.0

    X.append(img)

    label = row[classes].values.astype(np.float32)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("Formato de X: ", X.shape)
print("Formato de y: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

print("Treino: ", X_train.shape, y_train.shape)
print("Teste: ", X_test.shape, y_test.shape)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train_flat)
X_test_std = sc.transform(X_test_flat)

tf.keras.backend.clear_session()

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_std.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu', name='oculta1'),
    tf.keras.layers.Dense(64, activation='relu', name='oculta2'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='saida')
])

modelo.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

modelo.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

historico = modelo.fit(X_train_std, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

plt.figure(figsize=(8, 5))
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Função Perda')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'], loc='upper right')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia')
plt.ylabel('Accuracy')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()

saida_predida_prob = modelo.predict(X_test_std)
saida_predida = np.argmax(saida_predida_prob, axis=1)
saida_real = np.argmax(y_test, axis=1)

print('Acurácia:', accuracy_score(saida_real, saida_predida))
print('Precisão macro:', precision_score(saida_real, saida_predida, average='macro', zero_division=0))
print('Recall macro:', recall_score(saida_real, saida_predida, average='macro', zero_division=0))
print('F1-score macro:', f1_score(saida_real, saida_predida, average='macro', zero_division=0))

print("\nRelatório de Classificação:\n")
print(classification_report(saida_real, saida_predida, target_names=classes, zero_division=0))

cm = confusion_matrix(saida_real, saida_predida)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Matriz de Confusão')
plt.show()