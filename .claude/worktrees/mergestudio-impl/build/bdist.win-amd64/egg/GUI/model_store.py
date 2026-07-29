from __future__ import annotations

import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class ModelSchema:
    defaults: dict[str, Any]
    choices: dict[str, list[Any]]
    labels: dict[str, str]
    help: dict[str, str]
    types: dict[str, str]
    order: list[str]


def get_model_schema(model_class: str) -> ModelSchema:
    """Return schema for a given model class.

    Runtime rule: ONLY read from gui/schema_registry.json.
    GUI will not parse project/model source, will not cache, and will not overwrite the registry.
    """
    repo_root = Path(__file__).resolve().parents[1]
    static = _read_static_schema(repo_root, model_class)
    if static is None:
        return ModelSchema(defaults={}, choices={}, labels={}, help={}, types={}, order=[])
    return static


def _read_static_schema(repo_root: Path, model_class: str) -> ModelSchema | None:
    path = Path(repo_root) / "gui" / "data.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    raw = data.get(model_class)
    if not isinstance(raw, dict):
        return None

    defaults = raw.get("defaults", {})
    choices = raw.get("choices", {})
    labels = raw.get("labels", {})
    help_text = raw.get("help", {})
    types = raw.get("types", {})
    order = raw.get("order", [])

    if not isinstance(defaults, dict):
        defaults = {}
    if not isinstance(choices, dict):
        choices = {}
    if not isinstance(labels, dict):
        labels = {}
    if not isinstance(help_text, dict):
        help_text = {}
    if not isinstance(types, dict):
        types = {}
    if not isinstance(order, list):
        order = []

    # Ensure choices values are lists.
    fixed_choices: dict[str, list[Any]] = {}
    for k, v in choices.items():
        if isinstance(v, list):
            fixed_choices[str(k)] = v
        elif isinstance(v, tuple):
            fixed_choices[str(k)] = list(v)

    return ModelSchema(
        defaults={str(k): v for k, v in defaults.items()},
        choices=fixed_choices,
        labels={str(k): str(v) for k, v in labels.items()},
        help={str(k): str(v) for k, v in help_text.items()},
        types={str(k): str(v) for k, v in types.items()},
        order=[str(x) for x in order],
    )


@dataclass(frozen=True)
class ModelInfo:
    base_name: str
    model_class: str
    data_path: Path
    iter: int

    @property
    def display_name(self) -> str:
        return f"{self.base_name}_{self.model_class}"


def _extract_base_and_class_from_data_filename(filename: str) -> tuple[str, str] | None:
    # Expected: <base>_<class>_data.dat where <base> may contain underscores.
    # We detect by suffix "_data.dat" and parse the last "_<class>" part.
    if not filename.endswith("_data.dat"):
        return None
    stem = filename[: -len("_data.dat")]
    if "_" not in stem:
        return None
    base, model_class = stem.rsplit("_", 1)
    if not base or not model_class:
        return None
    return base, model_class


def scan_models(model_dir: Path) -> list[ModelInfo]:
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.exists():
        return []

    out: list[ModelInfo] = []
    for p in sorted(model_dir.glob("*_data.dat")):
        parsed = _extract_base_and_class_from_data_filename(p.name)
        if parsed is None:
            continue
        base, model_class = parsed
        iter_num = 0
        try:
            data = pickle.loads(p.read_bytes())
            if isinstance(data, dict):
                iter_num = int(data.get("iter", 0) or 0)
        except Exception:
            iter_num = 0

        out.append(ModelInfo(base_name=base, model_class=model_class, data_path=p, iter=iter_num))

    # Sort newest first by mtime.
    out.sort(key=lambda x: x.data_path.stat().st_mtime if x.data_path.exists() else 0.0, reverse=True)
    return out


def read_model_data(data_path: Path) -> dict[str, Any]:
    data_path = Path(data_path)
    data = pickle.loads(data_path.read_bytes())
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected model data type: {type(data)}")
    return data


def read_options(data_path: Path) -> dict[str, Any]:
    data = read_model_data(data_path)
    options = data.get("options", {})
    if not isinstance(options, dict):
        return {}
    return options


def default_options_path(model_dir: Path, model_class: str) -> Path:
    return Path(model_dir) / f"{model_class}_default_options.dat"


def read_default_options(model_dir: Path, model_class: str) -> dict[str, Any]:
    p = default_options_path(model_dir, model_class)
    if not p.exists():
        # Fall back to parsing model source for defaults (cached).
        return get_model_schema(model_class).defaults
    try:
        data = pickle.loads(p.read_bytes())
        file_defaults = data if isinstance(data, dict) else {}
        if file_defaults:
            return file_defaults
        return get_model_schema(model_class).defaults
    except Exception:
        return get_model_schema(model_class).defaults


def read_model_choices(model_class: str) -> dict[str, list[Any]]:
    """Best-effort choices list (for normal-mode dropdowns)."""
    return get_model_schema(model_class).choices


def read_model_help(model_class: str) -> dict[str, str]:
    return get_model_schema(model_class).help


def read_model_labels(model_class: str) -> dict[str, str]:
    return get_model_schema(model_class).labels


def read_model_types(model_class: str) -> dict[str, str]:
    return get_model_schema(model_class).types


def read_model_order(model_class: str) -> list[str]:
    return get_model_schema(model_class).order


def backup_file(path: Path) -> Path:
    path = Path(path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(path.name + f".bak.{ts}")
    shutil.copy2(path, backup_path)
    return backup_path


def write_options(data_path: Path, new_options: dict[str, Any], *, make_backup: bool = True) -> Path | None:
    """Write options back into *_data.dat safely.

    Returns backup path if created.
    """
    data_path = Path(data_path)
    if make_backup:
        backup = backup_file(data_path)
    else:
        backup = None

    data = read_model_data(data_path)
    data["options"] = dict(new_options)
    data_path.write_bytes(pickle.dumps(data))

    # Re-load for quick validation.
    _ = read_model_data(data_path)
    return backup
