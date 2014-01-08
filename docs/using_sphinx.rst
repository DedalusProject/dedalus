Using Sphinx with Dedalus
*******************************************

Here are extremely rough notes on using ``sphinx`` with ``Dedalus``.

To install ``sphinx``, do::

     pip3 install sphinx

(for python3 install).

To setup ``sphinx`` on your system, run the following at the terminal in the ``sphinx_docs`` directory::

     sphinx-quickstart

use default answers for everything except the following question::

    autodoc: automatically insert docstrings from modules (y/N) [n]: y

Then edit ``conf.py`` to point to your ``Dedalus`` directories that
you want to document.

Next, create a ``.rst`` file that pulls in the documentation
(see ``trasforms.rst`` for an example) and include it in the
``index.rst`` file.  Finally, run::

    make html

to creat html documentation in ``_build/html`` (can also do 
``make latex`` though we haven't explored this yet).  See your work by
opening ``_build/html/index.html``.
