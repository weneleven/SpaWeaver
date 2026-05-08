from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
