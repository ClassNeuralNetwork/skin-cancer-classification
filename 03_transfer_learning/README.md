# Etapa 03 — Transfer learning e DeepSMOTE

Experimentos com InceptionV3, DenseNet169, DenseNet201 e EfficientNet, combinados com crop pela máscara, Dull Razor, segmentação, oversampling, undersampling e DeepSMOTE.

- `metodologia_deep_skin/`: variações da metodologia de pré-processamento.
- `deep_smote/`: geração do conjunto sintético, treinamento e avaliação.
- `scripts/`: scripts auxiliares para produzir figuras metodológicas.

Para executar o DeepSMOTE em outra máquina, siga o roteiro em `deep_smote/README.md`. Ele inclui a preparação do Git LFS, a instalação do ambiente e uma verificação rápida da workstation.
