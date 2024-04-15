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
    'autoapi.extension',
    'sphinx_copybutton',
    'nbsphinx',
]
add_module_names = False
autoclass_content = "both"
templates_path = ['_templates']
autoapi_dirs = ['../step']
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*_script.rst"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = True
autodoc_typehints = "signature"
copybutton_selector = 'div.nbinput.container div.input_area div[class*=highlight] > pre'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme = "furo"

html_theme_options = {
    "source_repository": "https://github.com/SGGb0nd/step",
    "source_branch": "main",
    "source_directory": "docs/",
}

# nbsphinx_execute_arguments = [
#     "--InlineBackend.figure_formats={'svg', 'pdf'}",
# ]

nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

html_css_files = ["css/custom.css"]
def setup(app):
    app.add_css_file('css/custom.css')

def skip_member(app, what, name, obj, skip, options):
    # conditional breakpoint here
    return skip

def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_member)