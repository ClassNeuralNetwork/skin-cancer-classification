# Modelos

Modelos treinados são agrupados pela etapa que os produziu:

```text
models/
├── inception_v3/
├── cnn_resnet/
├── transfer_learning/
└── deep_smote/
```

Arquivos `.keras` e `.h5` são artefatos binários e devem ser armazenados com Git LFS. O nome do arquivo deve identificar arquitetura, pré-processamento e técnica de balanceamento sempre que possível.

Evite nomes genéricos como `model.keras` e checkpoints sem identificação do experimento.
