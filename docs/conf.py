# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"opengenome"
copyright = u"2025, Lu Hou Yang"
author = u"Lu Hou Yang"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_css_files = ["css/custom.css"]
html_theme_options = {
    "logo": {
        "link": "https://open-genome-project.org"
    },
    "icon_links": [{
        "name": "GitHub",
        "url": "https://github.com/luhouyang/opengenome.git",
        "icon": "fab fa-github",
        "type": "fontawesome",
    }],
    "external_links": [
        {
            "name": "website",
            "url": "https://open-genome-project.org"
        },
    ]
}
