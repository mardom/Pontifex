import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Pontifex'
copyright = '2026, The Pontifex Team & DESC Collaboration'
author = 'The Pontifex Team'
version = '2.1.0'
release = '2.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'autoapi.extension',
]

autoapi_dirs = ['../src/pontifex']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
