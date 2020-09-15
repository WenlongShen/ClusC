#!/usr/bin/env python
# coding: utf-8

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clusc",
    version="0.0.1",
    author="Wenlong Shen",
    author_email="shenwl1988@gmail.com",
    description="A toolkit for clustering and visualization of Hi-C sub-networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenlongShen/ClusC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)