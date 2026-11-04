"""Utilitários de caminhos compartilhados pelos experimentos DeepSMOTE."""

from pathlib import Path


def find_project_root() -> Path:
    """Localiza a raiz do repositório sem depender do diretório de execução."""
    module_path = Path(__file__).resolve()

    for candidate in (module_path.parent, *module_path.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "03_transfer_learning" / "deep_smote").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Não foi possível localizar a raiz do projeto a partir de "
        f"{module_path}."
    )


PROJECT_ROOT = find_project_root()
LEGACY_DEEPSMOTE_DIR = PROJECT_ROOT / "03_transfer_learning" / "deep_smote"


def to_project_relative(path: str | Path) -> str:
    """Serializa um caminho dentro do repositório de forma portável."""
    resolved = Path(path).expanduser().resolve()

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"O caminho precisa estar dentro do projeto: {resolved}"
        ) from exc


def resolve_project_path(path: str | Path) -> str:
    """Resolve caminhos novos e caminhos legados dos CSVs DeepSMOTE."""
    value = Path(path).expanduser()

    if value.is_absolute():
        return str(value.resolve())

    candidates = (
        PROJECT_ROOT / value,
        LEGACY_DEEPSMOTE_DIR / value,
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((PROJECT_ROOT / value).resolve())
