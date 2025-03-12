import datetime

import temporaldata

author = "neuro-galaxy Team"
project = "temporaldata"
version = temporaldata.__version__
copyright = f"{datetime.datetime.now().year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_inline_tabs",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]

add_module_names = False
autodoc_member_order = "bysource"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
}

myst_enable_extensions = [
    "html_admonition",
    "html_image",
]

pygments_style = "default"
html_copy_source = False
html_show_sourcelink = True
html_logo = "_static/temporaldata_logo.png"
html_favicon = "_static/temporaldata_logo.png"
