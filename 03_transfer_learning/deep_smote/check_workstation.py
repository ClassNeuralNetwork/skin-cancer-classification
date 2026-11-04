"""Verifica se a workstation está pronta para executar o DeepSMOTE."""

from importlib.util import find_spec
from pathlib import Path
import shutil
import sys
import tempfile

from project_paths import PROJECT_ROOT


REQUIRED_MODULES = {
    "cv2": "opencv-python-headless",
    "keras": "keras",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "tensorflow": "tensorflow",
}


def count_files(directory: Path, patterns: tuple[str, ...]) -> int:
    return sum(
        1
        for pattern in patterns
        for path in directory.glob(pattern)
        if path.is_file()
    )


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(64).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def main() -> int:
    errors = []
    warnings = []

    print("Raiz do projeto:", PROJECT_ROOT)

    missing_modules = [
        package
        for module, package in REQUIRED_MODULES.items()
        if find_spec(module) is None
    ]
    if missing_modules:
        errors.append(
            "Dependências ausentes: " + ", ".join(sorted(missing_modules))
        )

    if shutil.which("git-lfs") is None and shutil.which("git") is not None:
        warnings.append(
            "Git LFS não foi localizado. Instale-o e execute `git lfs pull`."
        )

    data_dir = PROJECT_ROOT / "data"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    csv_path = data_dir / "GroundTruth.csv"
    output_dir = data_dir / "deepsmote" / "generated_images"

    if not csv_path.is_file():
        errors.append(f"Metadados não encontrados: {csv_path}")

    image_count = count_files(images_dir, ("*.jpg", "*.jpeg", "*.png"))
    mask_count = count_files(masks_dir, ("*.jpg", "*.jpeg", "*.png"))

    if image_count == 0:
        errors.append(
            f"Nenhuma imagem encontrada em {images_dir}. Execute `git lfs pull`."
        )
    if mask_count == 0:
        errors.append(
            f"Nenhuma máscara encontrada em {masks_dir}. Execute `git lfs pull`."
        )

    samples = [
        next(
            (path for path in images_dir.iterdir() if path.is_file()),
            None,
        )
        if images_dir.is_dir()
        else None,
        next(
            (path for path in masks_dir.iterdir() if path.is_file()),
            None,
        )
        if masks_dir.is_dir()
        else None,
    ]
    if any(path is not None and is_lfs_pointer(path) for path in samples):
        errors.append(
            "Os dados ainda são ponteiros do Git LFS. Execute `git lfs pull`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir):
            pass
    except OSError as exc:
        errors.append(f"A pasta de saída não permite escrita: {exc}")

    free_gib = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    if free_gib < 20:
        warnings.append(
            f"Somente {free_gib:.1f} GiB livres; recomenda-se ao menos 20 GiB."
        )

    print("Imagens originais:", image_count)
    print("Máscaras:", mask_count)
    print("Pasta de saída:", output_dir)
    print(f"Espaço livre: {free_gib:.1f} GiB")

    if find_spec("tensorflow") is not None:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        print("GPUs reconhecidas pelo TensorFlow:", gpus)
        if not gpus:
            warnings.append(
                "O TensorFlow não reconheceu uma GPU; a execução será muito lenta."
            )

    for warning in warnings:
        print("AVISO:", warning)

    if errors:
        for error in errors:
            print("ERRO:", error)
        return 1

    print("Workstation pronta para executar o gerador DeepSMOTE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
