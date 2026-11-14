# Validação cruzada das arquiteturas

Esta pasta reúne os experimentos individuais da DenseNet169, DenseNet201 e EfficientNetB4. As três arquiteturas usam validação cruzada estratificada em cinco folds e seguem o mesmo protocolo de preparação dos dados.

## Metodologia

- Os folds são formados somente com `data/deepsmote/train_real_split.csv`; `test_real_split.csv` fica reservado para a avaliação final.
- Cada imagem real participa da validação uma vez e integra o treino nos outros quatro folds.
- O oversampling é aplicado somente depois da divisão, sobre o subconjunto de treino. As classes minoritárias são replicadas até alcançarem o tamanho da maior classe do fold.
- A validação mantém a distribuição original. Não são usados pesos de classe nem imagens DeepSMOTE.
- Os backbones partem dos pesos ImageNet, permanecem congelados durante esta etapa e compartilham a mesma cabeça classificadora e as mesmas transformações de aumento de dados.

## Notebooks

- `densenet169_cross_validation.ipynb`
- `densenet201_cross_validation.ipynb`
- `efficientnetb4_cross_validation.ipynb`

Cada notebook salva os resultados em `results/cross_validation/<arquitetura>/`: atribuição dos folds, distribuição de classes, históricos de treinamento, métricas, predições out-of-fold e matriz de confusão agregada.

Os modelos de cada fold não são gravados por padrão, pois ocupariam centenas de megabytes. Para mantê-los, use `save_fold_models=True` na configuração do experimento.

O ensemble que combina as três arquiteturas está em `04_ensembles/cross_validation/ensemble_3_cross_validation.ipynb`.
