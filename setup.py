#!/usr/bin/env python

import setuptools

setuptools.setup(
        name="lif_summary",
        version="2.1",
        author="Oliver Schneider",
        author_email="ghilorevilo@posteo.de",
        description="automated extraction of microscope images/ videos from .lif-files",
        long_description="",
        url="https://github.com/ilorevilo/lif_summary",
        py_modules=['lif_summary'],
        install_requires=[
            'numpy>=1.19.5',
            'colorama>=0.4.4',
            'tifffile>=2021.4.8',
            'imageio-ffmpeg>=0.4.3',
            'scikit_image>=0.18.1',
            'tqdm>=4.61.1',
            'Pillow>=8.3.1',
            'readlif>=0.6.2',
            'opencv-python>=4.5.1'
        ],        
        python_requires='>=3.8',
        entry_points={
            'console_scripts': ['lifsum=lif_summary:main',
                                'lif_summary=lif_summary:main'],
            },
        )