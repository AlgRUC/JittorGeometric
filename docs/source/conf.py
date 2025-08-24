import os
import sys

# Add the project root directory to Python path for autodoc
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JittorGeometric'
copyright = '2025, JittorGeometric Team'
author = 'JittorGeometric Team'
release = '2.0.0'
version = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
]

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary configuration
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Template paths
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Language
language = 'en'

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'recommonmark',
}

# Master document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Custom logo
html_logo = '../../assets/JittorGeometric_logo.png'
html_favicon = None

# HTML title
html_title = f'{project} v{version} documentation'

# Output directory for ReadTheDocs
html_build_dir = os.environ.get('READTHEDOCS_OUTPUT', 'build/html')

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {}
latex_documents = [
    (master_doc, 'JittorGeometric.tex', 'JittorGeometric Documentation',
     'JittorGeometric Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'jittorgeometric', 'JittorGeometric Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'JittorGeometric', 'JittorGeometric Documentation',
     author, 'JittorGeometric', 'Machine learning on graph data with Jittor.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_use_rtype = True