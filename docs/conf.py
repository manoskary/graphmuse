"""Sphinx configuration for the GraphMuse documentation."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata

# -- Path setup --------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -- Project information -----------------------------------------------------

project = "GraphMuse"
author = "GraphMuse Developers"
copyright = f"{datetime.now():%Y}, {author}"

# Fallback to package version if available; otherwise, use the latest tag.
try:
    release = importlib_metadata.version("graphmuse")
except Exception:
    release = "0.0.5"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Avoid importing heavy dependencies during doc builds (especially on RTD).
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_sparse",
    "torch_scatter",
    "torch_cluster",
    "pyg_lib",
    "partitura",
    "psutil",
    "graphmuse.samplers.csamplers",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

try:
    import sphinx_rtd_theme  # type: ignore
except ImportError:  # pragma: no cover - local builds without theme
    html_theme = "alabaster"
else:
    html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}
