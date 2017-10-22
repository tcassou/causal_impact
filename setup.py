# -*- coding: utf-8 -*-
from distutils.core import setup


setup(
    name='causal_impact',
    packages=['causal_impact'],
    version='1.0.0',
    description='Python package for causal inference using Bayesian structural time-series models.',
    url='https://github.com/tcassou/causal_impact',
    download_url='https://github.com/tcassou/causal_impact/archive/1.0.0.tar.gz',
    keywords=['bayesian', 'structural', 'time-series', 'causal', 'impact', 'python', 'inference'],
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=[
        'numpy>=1.11.3',
        'scipy>=0.18.1',
        'matplotlib>=1.5.3',
        'statsmodels>=0.8.0',
    ],
)
