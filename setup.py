# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib import Path
from setuptools import setup, find_packages

# ...
# Read library version into '__version__' variable
path = Path(__file__).parent / 'manapy' / 'version.py'
exec(path.read_text())
# ...

NAME    = 'manapy'
VERSION = __version__
AUTHOR  = 'Imad Kissami'
EMAIL   = 'imad.kissami@gmail.ma'
URL     = 'https://github.com/pyccel/manapy'
DESCR   = 'TODO.'
KEYWORDS = ['math']
LICENSE = "LICENSE"

setup_args = dict(
    name                 = NAME,
    version              = VERSION,
    description          = DESCR,
    #long_description     = open('README.rst').read(),
    author               = AUTHOR,
    author_email         = EMAIL,
    license              = LICENSE,
    keywords             = KEYWORDS,
#    url                  = URL,
)

# ...
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

# Dependencies
install_requires = [
    'numpy',
    'meshio<4',
    'mpi4py',
    'mgmetis',
    'numba>0.4',
    'scipy',
    #'timeit',
    ]

def setup_package():
    setup(packages=packages, \
          include_package_data=True, \
          install_requires=install_requires, \
         **setup_args)

if __name__ == "__main__":
    setup_package()
