"""Setup module for Iris."""

from setuptools import setup, find_packages

setup(
        name='IrisParseNet',
        version='1.0',
        description='Data-driven iris segmentation and localization using machine learning.',

        author='Caiyong Wang',
        author_email='wangcaiyong2017@ia.ac.cn',

        packages=find_packages(exclude=[]),
        python_requires='>=3.6',
        install_requires=[
            'coloredlogs',
            'numpy',
            'opencv-python',
            'torch',
            'torchvision',
            'imageio',  # http://imageio.github.io/
            'scipy',
        ],
)
