import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shap

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

def model_predict(X):
    return np.argmax(modelo.predict(X), axis=1)

# from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "GroundTruth.csv")

IMG_SIZE = (64, 64)
NUM_CLASSES = 7
classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

metadata = pd.read_csv(CSV_PATH)

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

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)

# y em formato de índice para estratificação e SMOTE
y_indices = np.argmax(y, axis=1)

X_train, X_test, y_train_idx, y_test_idx = train_test_split(
    X,
    y_indices,
    test_size=0.2,
    random_state=42,
    stratify=y_indices
)

print("Treino:", X_train.shape, y_train_idx.shape)
print("Teste:", X_test.shape, y_test_idx.shape)

# Flatten
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Padronização
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_flat)
X_test_std = scaler.transform(X_test_flat)

print("\nDistribuição antes do SMOTE:")
print(pd.Series(y_train_idx).value_counts().sort_index())

# SMOTE SOMENTE no treino
smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote_idx = smote.fit_resample(X_train_std, y_train_idx)

print("\nDistribuição depois do SMOTE:")
print(pd.Series(y_train_smote_idx).value_counts().sort_index())

y_train_smote = tf.keras.utils.to_categorical(y_train_smote_idx, num_classes=NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test_idx, num_classes=NUM_CLASSES)

tf.keras.backend.clear_session()

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_smote.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01), name='oculta1'),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01), name='oculta2'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='saida')
])

modelo.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

modelo.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights='True')

historico = modelo.fit(
    X_train_smote,
    y_train_smote,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
    # callbacks=[early_stopping]
)

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
saida_real = y_test_idx

print('Acurácia:', accuracy_score(saida_real, saida_predida))
print('Precisão weighted:', precision_score(saida_real, saida_predida, average='weighted', zero_division=0))
print('Recall weighted:', recall_score(saida_real, saida_predida, average='weighted', zero_division=0))
print('F1-score weighted:', f1_score(saida_real, saida_predida, average='weighted', zero_division=0))

print("\nRelatório de Classificação:\n")
print(classification_report(saida_real, saida_predida, target_names=classes, zero_division=0))

cm = confusion_matrix(saida_real, saida_predida)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Matriz de Confusão')
plt.show()

explainer = shap.KernelExplainer(model_predict, X_test_std)

shap_values = explainer.shap_values(X_test_std[0:100])

shap.summary_plot(shap_values, X_test_std, feature_names=)