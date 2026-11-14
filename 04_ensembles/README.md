# Etapa 04 — Ensembles, TTA e explicabilidade

Etapa final do TCC, dedicada à combinação dos modelos, avaliação com TTA e multi-crop e geração de visualizações Grad-CAM.

Os resultados devem ser salvos em `../results/ensembles/`.

## Validação cruzada

O notebook `cross_validation/ensemble_3_cross_validation.ipynb` treina ou reaproveita os resultados out-of-fold da DenseNet169, DenseNet201 e EfficientNetB4. As probabilidades dos três modelos são combinadas por *soft voting*, com pesos iguais definidos antes da avaliação.

Os arquivos desse experimento são gravados em `../results/cross_validation/ensemble-3-soft-voting/`.
