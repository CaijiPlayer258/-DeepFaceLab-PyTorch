"""Launcher entrypoint for the new Tk GUI.

All GUI implementation lives in the `gui/` package, per project constraint:
- Do NOT modify upstream/original DeepFaceLab code.
- Root-level gui.py stays a thin launcher.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
import types


def _ensure_repo_root_on_syspath() -> None:
	# When launching via double-click or from other cwd, ensure imports work.
	repo_root = Path(__file__).resolve().parent
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	os.chdir(str(repo_root))


def main() -> int:
	_ensure_repo_root_on_syspath()
	repo_root = Path(__file__).resolve().parent
	app_path = repo_root / "gui" / "app.py"
	if not app_path.exists():
		raise FileNotFoundError(f"Missing GUI module: {app_path}")

	# Create a synthetic package named 'gui' so that gui/app.py can use
	# relative imports like `from . import model_store`.
	pkg_name = "gui"
	if pkg_name not in sys.modules or not getattr(sys.modules[pkg_name], "__path__", None):
		pkg = types.ModuleType(pkg_name)
		pkg.__path__ = [str((repo_root / "gui").resolve())]  # type: ignore[attr-defined]
		pkg.__package__ = pkg_name
		sys.modules[pkg_name] = pkg

	spec = importlib.util.spec_from_file_location("gui.app", app_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Unable to load module spec for {app_path}")

	mod = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = mod
	spec.loader.exec_module(mod)

	gui_main = getattr(mod, "main", None)
	if not callable(gui_main):
		raise AttributeError("gui/app.py must define callable main()")

	return int(gui_main())


if __name__ == "__main__":
	raise SystemExit(main())