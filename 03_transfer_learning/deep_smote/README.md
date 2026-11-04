# DeepSMOTE na workstation

O gerador principal é `01_generate_deepsmote_dataset.ipynb`. A versão equivalente para execução direta é `deepsmote_images.py`.

## Preparação após clonar

Na raiz do repositório:

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python 03_transfer_learning/deep_smote/check_workstation.py
```

O último comando deve terminar com `Workstation pronta para executar o gerador DeepSMOTE.` antes de iniciar o processamento longo.

## Execução

Pelo terminal, a partir de qualquer diretório:

```bash
python 03_transfer_learning/deep_smote/deepsmote_images.py
```

Também é possível abrir e executar o notebook `01_generate_deepsmote_dataset.ipynb` no VS Code.

As imagens sintéticas são salvas automaticamente em `data/deepsmote/generated_images/<classe>/`. O formato é PNG sem perdas. O CSV `data/deepsmote/metadata_deepsmote.csv` registra caminhos relativos à raiz do projeto, portanto permanece válido depois de clonar o repositório em outro computador.

Ao final, `data/deepsmote/generation_config.json` registra a configuração e as contagens da execução para permitir sua reprodução.

Os modelos do autoencoder gerados com a configuração atual ficam em `models/deep_smote/` e têm `crop_256_latent512` no nome, evitando sobrescrever os modelos antigos de resolução diferente.

Se a GPU não comportar `BATCH_SIZE = 16`, reduza o valor para `8` ou `4` na seção de configurações do gerador.
