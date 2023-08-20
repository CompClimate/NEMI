# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

# Generate the api
if sys.platform in ["linux", "darwin"]:
    subprocess.check_output(["make", "generate-api"], cwd=os.path.dirname(os.path.abspath(__file__)))
else:
    subprocess.check_output(["make.bat", "generate-api"], cwd=os.path.dirname(os.path.abspath(__file__)))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NEMI'
copyright = '2023, Maike Sonnewald'
author = 'Maike Sonnewald'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'nbsphinx',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'member-order': 'bysource',
    'members': None,
    'exclude-members': '__all__'
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']