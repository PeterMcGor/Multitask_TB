# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:06:46 2018

@author: peter
"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['simpleitk', 'tensorlayer', 'dltk', 'configargparse']

setup(
    name='DIAG_package',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=['pandas', 'tensorflow', 'numpy']
)
