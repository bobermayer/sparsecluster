=============
SparseCluster
=============

sparsecluster is a python package to perform hierarchical agglomerative clustering
on approximate nearest neighbor graphs. it's basically a wrapper around pynndescent
and gbbs

------------
installation
------------

clone this repo and load all associated submodules. installing python bindings for `gbbs` can be a bit tricky, below is a hacky workaround. This will probably be resolved in the separate [ParHAC](https://github.com/ParAlg/ParHAC) repo.

.. code:: bash

    git clone git@github.com:bobermayer/sparsecluster.git
    cd sparsecluster
    git submodule update --init
    cd gbbs
    git submodule update --init
    bazel build //...
    bazel build //pybindings:gbbs_lib.so
    cd ..
    pip install .
    cp gbbs/bazel-bin/pybindings/gbbs_lib.so /path/to/python/site-packages/gbbs_lib.cpython-??-x86_64-linux-gnu.so 

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
* libprotobuf

all of which should be pip or conda installable. 

in addition, a custom fork of `pynndescent` is required:

.. code:: bash

    git clone git@github.com:bobermayer/pynndescentSC.git
    cd pynndescentSC
    pip install .
