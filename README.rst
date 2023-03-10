=============
SparseCluster
=============

sparsecluster is a python package to perform hierarchical agglomerative clustering
on approximate nearest neighbor graphs. it's basically a wrapper around pynndescent
and gbbs

------------
installation
------------

clone this repo and load all associated submodules

.. code:: bash
    git clone git@github.com:bobermayer/sparsecluster.git
    cd sparsecluster/gbbs
    git submodule update --init
    cd ..
    pip install .

-----
usage
-----

call signature is essentially similar to `fastcluster`:

.. code:: python
    Z = sparsecluster.linkage(X, metric='euclidean', method='single', n_neighbors=50, n_backup=10, n_jobs=2, ...)

where `X` is a (possibly sparse) matrix of dimensions `(n_samples, n_features)`. 

------------
dependencies
------------

* numpy
* scipy
* scikit-learn >= 0.22
* numba >= 0.51
* bazel

all of which should be pip or conda installable. 

in addition, a custom fork of `pynndescent` is required:

.. code:: bash
    git clone git@github.com:bobermayer/pynndescentSC.git
    cd pynndescentSC
    pip install .
