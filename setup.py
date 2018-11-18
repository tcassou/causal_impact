# -*- coding: utf-8 -*-
from distutils.core import setup


version = '1.0.4'

setup(
    name='causal_impact',
    packages=['causal_impact'],
    version=version,
    description='Python package for causal inference using Bayesian structural time-series models.',
    url='https://github.com/tcassou/causal_impact',
    download_url='https://github.com/tcassou/causal_impact/archive/{}.tar.gz'.format(version),
    keywords=['bayesian', 'structural', 'time-series', 'causal', 'impact', 'python', 'inference'],
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=[
        'numpy>=1.14.3',
        'scipy>=0.18.1',
        'matplotlib>=1.5.3',
        'statsmodels>=0.9.0',
    ],
)
