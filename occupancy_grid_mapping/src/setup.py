#!/usr/bin/env python
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension("my_occ_grid_map", ["my_occ_grid_map.py"]), Extension("my_occ_grid_map_cell_by_cell", ["my_occ_grid_map_cell_by_cell.py"])]

setup(ext_modules=cythonize(ext_modules))