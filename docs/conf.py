from pathlib import Path
import sys
import types


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

spaweaver_package = types.ModuleType("SpaWeaver")
spaweaver_package.__path__ = [str(repo_root)]
sys.modules.setdefault("SpaWeaver", spaweaver_package)

project = "SpaWeaver"
author = "SpaWeaver contributors"
copyright = "2026, SpaWeaver contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

autodoc_mock_imports = [
    "anndata",
    "cv2",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "scanpy",
    "scipy",
    "skimage",
    "sklearn",
    "torch",
    "torchvision",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
