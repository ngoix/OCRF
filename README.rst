.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Coveralls|_ |CircleCI|_ |Python27|_ |Python35|_ |PyPi|_ 

.. |Travis| image:: https://api.travis-ci.org/scikit-learn/scikit-learn.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn/scikit-learn

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/github/scikit-learn/scikit-learn?branch=master&svg=true
.. _AppVeyor: https://ci.appveyor.com/project/sklearn-ci/scikit-learn/history

.. |Coveralls| image:: https://coveralls.io/repos/scikit-learn/scikit-learn/badge.svg?branch=master&service=github
.. _Coveralls: https://coveralls.io/r/scikit-learn/scikit-learn

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn/scikit-learn/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn/scikit-learn

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/scikit-learn

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

.. |PyPi| image:: https://badge.fury.io/py/scikit-learn.svg
.. _PyPi: https://badge.fury.io/py/scikit-learn

One Class Splitting Criteria for Random Forests
============
This repository provide the code corresponding to the article `One Class Splitting Criteria for Random Forests <https://arxiv.org/pdf/1611.01971v3.pdf>`_, and other anomaly detection algorithms. 

Abstract
=======

Random Forests (RFs) are strong machine learning tools for classification and regression.
However, they remain supervised algorithms, and no extension of RFs to the one-class setting has
been proposed, except for techniques based on second-class sampling. This work fills this gap
by proposing a natural methodology to extend standard splitting criteria to the one-class setting,
structurally generalizing RFs to one-class classification.  An extensive benchmark of seven
state-of-the-art anomaly detection algorithms is also presented. This empirically demonstrates
the relevance of our approach.

Install
=======

The implementation is based on a fork of scikit-learn. To have a working version of both scikit-learn and OCRF scikit-learn one can use conda to create a virtual environment specific to OCRF while keeping the original version of scikit-learn clean.

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py build_ext --inplace

and run your personal code inside the folder OCRF. To use OCRF outside of the OCRF folder change the environment variable PYTHONPATH or create a virtual environment with Conda.

Install with Conda
=======

First install conda `Conda <https://docs.continuum.io/anaconda/install>`_ and update it::
  
  conda update conda
  conda update --all
  
Then create a virtual environment for OCRF, activate it and install OCRF and its dependencies on the new virtual environment::

  conda create -n OCRF_env python=2.7 anaconda
  source activate OCRF_env
  conda install -n OCRF_env numpy scipy cython matplotlib

  git clone https://github.com/ngoix/OCRF

  cd OCRF
  pip install --upgrade pip
  pip install pyper
  python setup.py install
  cd ..

Now OCRF is installed. To check it run the script benchmark_oneclassrf.py::

  python benchmarks/benchmark_oneclassrf.py
  
To quit the environment and revert to the original scikit-learn use::

  source deactivate
  
To return to the OCRF environment use::
  
  source activate OCRF_env

scikit-learn
============

`scikit-learn <http://scikit-learn.org/>`_ is a Python module for machine learning built on top of
SciPy and distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the AUTHORS.rst file for a complete list of contributors.

It is currently maintained by a team of volunteers.

**Note** `scikit-learn` was previously referred to as `scikits.learn`.
