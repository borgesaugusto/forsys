# Configuration file for the Sphinx documentation builder.
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

project = 'ForSys'
author = 'Augusto Borges'

version = '0.4.0'

# -- General configuration

templates_path = ['_templates']

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'nbsphinx_link'
    # 'sphinx.ext.duration',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
# epub_show_urls = 'footnote'

autodoc_default_flags = ['members']
autosummary_generate = True
autosummary_imported_members = False
autoclass_content = "both"

nbsphinx_execute = 'never'

autodoc_mock_imports = ['numpy', 'scipy', "dataclasses", "warnings"]