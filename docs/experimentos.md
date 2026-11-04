# Catálogo de experimentos

Este documento relaciona a evolução do TCC aos arquivos do repositório. A pasta indica a etapa científica; o nome do arquivo descreve a arquitetura ou a variação experimental.

## 01 — MLP

| Arquivo | Experimento |
| --- | --- |
| `mlp.py` | baseline MLP com pixels normalizados |
| `mlp_pesos.py` | MLP com pesos de classe |
| `mlp_smote.py` | MLP com SMOTE e explicabilidade SHAP |
| `mlp_pesos_smote.py` | MLP com SMOTE e pesos de classe |

## 02 — CNN e ResNet50

| Grupo | Arquivos principais |
| --- | --- |
| CNN baseline | `cnn.py` |
| Balanceamento | `cnn_class_weights.ipynb`, `cnn_over_sampling.ipynb`, `cnn_smote.ipynb` |
| Transfer learning | `cnn_resnet50.ipynb`, `cnn_cw_resnet.ipynb`, `cnn_smote_2.ipynb` |
| Segmentação | `cnn_masks_only.ipynb`, `cnn_segmented_images.ipynb`, `visualizacao_segmentacao.ipynb` |
| Explicabilidade | `gradcam_resnet50.ipynb` |
| Consolidação visual | `matrizes.ipynb` |

## 03 — Transfer learning e DeepSMOTE

| Grupo | Arquivos principais |
| --- | --- |
| InceptionV3 | `inception_v3_baseline.py` |
| DenseNet169 | `densenet169.ipynb`, `densenet169_oversampling.ipynb`, `densenet169_undersampling.ipynb` |
| DenseNet201 | `densenet201.ipynb`, `densenet201_oversampling.ipynb`, `densenet201_deepsmote.ipynb` |
| EfficientNet | `efficientnet.ipynb`, `efficientnet_oversampling.ipynb` |
| Metodologia Deep Skin | `metodologia_deep_skin/` |
| Pipeline DeepSMOTE | `deep_smote/01_generate_deepsmote_dataset.ipynb` |
| Treino com DeepSMOTE salvo | notebooks `02_train_*` em `deep_smote/` |
| Avaliação DeepSMOTE | notebooks `avaliar_metricas_*` em `deep_smote/` |

## 04 — Ensembles e inferência

| Arquivo | Experimento |
| --- | --- |
| `ensemble_2.ipynb` | ensemble de dois modelos |
| `ensemble_3.ipynb` | ensemble de três modelos |
| `ensemble_3_multicrop.ipynb` | ensemble com multi-crop TTA |
| `ensemble_3_multicrop_gradcam.ipynb` | multi-crop TTA com Grad-CAM |
| `tta/avaliacao_tta.ipynb` | avaliação isolada de TTA |
| `deepsmote/ensemble_deepsmote.py` | ensemble dos modelos treinados com DeepSMOTE |

## Convenção para novos experimentos

Registre, no notebook ou em um JSON de métricas:

- arquitetura e tamanho da entrada;
- pré-processamento aplicado;
- técnica de balanceamento;
- seed e divisão dos dados;
- caminho do modelo treinado;
- accuracy, precision, recall e F1 macro e ponderados;
- configuração de TTA ou pesos do ensemble, quando aplicável.
