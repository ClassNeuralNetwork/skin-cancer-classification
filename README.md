# Classificação de câncer de pele

Repositório do Trabalho de Conclusão de Curso sobre classificação multiclasse de lesões de pele a partir de imagens dermatoscópicas.

O projeto está organizado de acordo com a evolução experimental do TCC: parte de uma MLP, avança para CNNs e transferência de aprendizado e termina com ensembles, TTA e técnicas de explicabilidade.

## Evolução dos experimentos

| Etapa | Objetivo | Principais técnicas |
| --- | --- | --- |
| [`01_mlp`](01_mlp/) | Construir o baseline | MLP, class weights e SMOTE |
| [`02_cnn_resnet`](02_cnn_resnet/) | Explorar informação espacial | CNN, ResNet50, máscaras, segmentação, SHAP e Grad-CAM |
| [`03_transfer_learning`](03_transfer_learning/) | Avaliar arquiteturas profundas | InceptionV3, DenseNet169, DenseNet201, EfficientNet, Dull Razor, crop e DeepSMOTE |
| [`04_ensembles`](04_ensembles/) | Combinar os melhores modelos | Ensemble ponderado, TTA, multi-crop e Grad-CAM |

O catálogo detalhado dos notebooks está em [`docs/experimentos.md`](docs/experimentos.md).

## Estrutura

```text
.
├── 01_mlp/                 # baseline e estratégias de balanceamento
├── 02_cnn_resnet/          # CNN, ResNet50, segmentação e explicabilidade
├── 03_transfer_learning/   # DenseNet, EfficientNet, InceptionV3 e DeepSMOTE
├── 04_ensembles/           # ensembles, TTA e multi-crop
├── data/                   # metadados, imagens, máscaras e dados derivados
├── models/                 # modelos agrupados por família experimental
├── results/                # figuras, métricas, predições e históricos
├── docs/                   # documentação dos experimentos
├── pyproject.toml
└── uv.lock
```

## Preparação do ambiente

O projeto utiliza Python 3.13 e `uv` para gerenciar as dependências.

```bash
uv sync
```

Os arquivos grandes são controlados com Git LFS. Após clonar o repositório, execute:

```bash
git lfs pull
```

## Dados

Os metadados ficam em `data/`, as imagens originais em `data/images/` e as máscaras em `data/masks/`. Informações adicionais estão em [`data/README.md`](data/README.md).

## Execução

Os scripts Python localizam a raiz do projeto automaticamente. Para os notebooks, abra o arquivo no VS Code ou Jupyter e use o próprio diretório do notebook como diretório de execução.

Modelos treinados são salvos em `models/` e resultados em `results/`. Evite salvar novos artefatos dentro das pastas de experimentos.

## Reprodutibilidade

- A divisão de treino e teste deve manter a mesma semente registrada em cada experimento.
- SMOTE, oversampling e undersampling devem ser aplicados somente ao conjunto de treino.
- Resultados comparáveis devem registrar arquitetura, pré-processamento, balanceamento, pesos do ensemble e seed.
- Notebooks executados devem ser associados aos modelos e métricas correspondentes no catálogo de experimentos.
