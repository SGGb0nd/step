# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'STEP'
copyright = '2024, Lounan Li'
author = 'Lounan Li'

sys.path.insert(0, os.path.abspath('..'))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    'sphinx.ext.autodoc',
    'nbsphinx',
]
add_module_names = False
autoclass_content = "both"
templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*_script.rst"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/SGGb0nd/step-dev",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "use_download_button": True,
    "use_fullscreen_button": False,
}

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

# html_css_files = ["css/custom.css"]
# def setup(app):
#     app.add_css_file('css/custom.css')
