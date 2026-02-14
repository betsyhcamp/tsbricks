"""Sphinx configuration."""

project = "tsbricks"
author = "Elizabeth H. Camp"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"
python_display_short_literal_types = True

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]