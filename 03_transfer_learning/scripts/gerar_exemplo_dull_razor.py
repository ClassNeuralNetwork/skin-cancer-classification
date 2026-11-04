import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path


# Aparencia global dos titulos, rotulos, legendas e arquivos exportados.
plt.rcParams.update({
    "font.family": "Arial",
    "legend.frameon": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
})


def carregar_entradas():
    """Carrega uma imagem e sua mascara, independentemente do diretorio atual."""
    raiz_projeto = Path(__file__).resolve().parents[1]
    # Exemplo com pelos escuros atravessando a lesao: a acao do Dull Razor
    # continua nitida mesmo quando a figura e reduzida em um slide.
    id_padrao = "ISIC_0025426"

    parser = ArgumentParser(description="Demonstra Dull Razor e recorte pela mascara.")
    parser.add_argument(
        "--imagem",
        type=Path,
        default=raiz_projeto / "data" / "images" / f"{id_padrao}.jpg",
    )
    parser.add_argument(
        "--mascara",
        type=Path,
        default=raiz_projeto / "data" / "masks" / f"{id_padrao}_segmentation.png",
    )
    args = parser.parse_args()

    imagem = cv2.imread(str(args.imagem))
    mascara = cv2.imread(str(args.mascara), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        parser.error(f"nao foi possivel ler a imagem: {args.imagem}")
    if mascara is None:
        parser.error(f"nao foi possivel ler a mascara: {args.mascara}")
    if imagem.shape[:2] != mascara.shape:
        parser.error(
            "imagem e mascara possuem dimensoes diferentes: "
            f"{imagem.shape[:2]} != {mascara.shape}"
        )
    if not np.any(mascara > 0):
        parser.error(f"a mascara nao possui pixels positivos: {args.mascara}")

    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), mascara


def aplicar_mascara_preta(imagem, mascara, opacidade=0.85):
    """Sobrepoe preto fora da lesao, preservando parte do contexto original."""
    resultado = imagem.copy()
    fora_da_lesao = mascara == 0
    resultado[fora_da_lesao] = (
        resultado[fora_da_lesao].astype(np.float32) * (1 - opacidade)
    ).astype(np.uint8)
    return resultado

# carregar imagem e máscara
img, mask = carregar_entradas()

# ---------------------------
# 1) DULL RAZOR
# ---------------------------
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

_, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
hair_mask = cv2.dilate(
    hair_mask,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    iterations=1,
)

img_dull = cv2.inpaint(
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    hair_mask,
    3,
    cv2.INPAINT_TELEA
)
img_dull = cv2.cvtColor(img_dull, cv2.COLOR_BGR2RGB)

# ---------------------------
# 2) BOUNDING BOX PELA MÁSCARA
# ---------------------------
ys, xs = np.where(mask > 0)

x_min, x_max = xs.min(), xs.max() + 1
y_min, y_max = ys.min(), ys.max() + 1

margin = 20
x_min = max(0, x_min - margin)
y_min = max(0, y_min - margin)
x_max = min(img.shape[1], x_max + margin)
y_max = min(img.shape[0], y_max + margin)

# recorte sem Dull Razor
crop_original = img[y_min:y_max, x_min:x_max]

# recorte com Dull Razor
crop_dull = img_dull[y_min:y_max, x_min:x_max]

# ---------------------------
# 3) VISUALIZAÇÃO
# ---------------------------
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
fig.patch.set_facecolor("none")
fig.patch.set_alpha(0)

for eixo in ax.flat:
    eixo.set_facecolor("none")

# linha 1: sem Dull Razor
ax[0,0].imshow(img)
ax[0,0].axis("off")

mask_vis = aplicar_mascara_preta(img, mask)
cv2.rectangle(mask_vis, (x_min, y_min), (x_max - 1, y_max - 1), (0,255,0), 4)

ax[0,1].imshow(mask_vis)
ax[0,1].axis("off")

ax[0,2].imshow(crop_original)
ax[0,2].axis("off")

# linha 2: com Dull Razor
ax[1,0].imshow(img_dull)
ax[1,0].axis("off")

mask_vis_dull = aplicar_mascara_preta(img_dull, mask)
cv2.rectangle(
    mask_vis_dull,
    (x_min, y_min),
    (x_max - 1, y_max - 1),
    (0,255,0),
    4,
)

ax[1,1].imshow(mask_vis_dull)
ax[1,1].axis("off")

ax[1,2].imshow(crop_dull)
ax[1,2].axis("off")

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.025, hspace=0.025)
saida = raiz_projeto / "results" / "figures" / "transfer_learning" / "exemplo_dull_razor_slide.png"
saida.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(saida, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
print(f"Figura salva em: {saida}")
plt.show()
