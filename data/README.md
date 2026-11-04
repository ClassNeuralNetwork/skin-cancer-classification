# Dados

Esta pasta contém os dados utilizados nos experimentos do TCC.

```text
data/
├── GroundTruth.csv
├── GroundTruth_masks.csv
├── images/       # imagens dermatoscópicas originais
├── masks/        # máscaras de segmentação
└── deepsmote/    # divisões, metadados, vetores e imagens sintéticas
```

Os arquivos grandes são controlados por Git LFS. A atribuição disponível no conjunto de imagens deve ser preservada em `images/ATTRIBUTION.txt`.

Arquivos de `deepsmote/` são dados derivados. Eles devem ser regenerados pelo notebook `03_transfer_learning/deep_smote/01_generate_deepsmote_dataset.ipynb` quando houver mudança na divisão dos dados ou no pré-processamento.

As imagens sintéticas são gravadas em `deepsmote/generated_images/`, em PNG sem perdas e separadas por classe. O gerador cria automaticamente as subpastas necessárias. Os caminhos registrados nos CSVs são relativos à raiz do projeto para funcionarem em máquinas diferentes.

Cada execução também grava `deepsmote/generation_config.json` com resolução, semente, dimensão latente, versões das bibliotecas e quantidade de imagens por classe.
