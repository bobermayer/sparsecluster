# Author: Benedikt Obermayer <benedikt.obermayer@bih-charite.de>
#
#

import time
import os
import sys
import numpy as np
import scipy.sparse
import pandas as pd

def _get_sparse_matrix_from_knn(knn_indices, knn_dists, min_dist=1.0e-10):

    nnodes, nneighbors = knn_indices.shape
    ii = np.repeat(np.arange(nnodes), nneighbors)
    jj = knn_indices.flatten("C")
    xx = knn_dists.flatten("C")
    ii, jj = np.minimum(ii, jj), np.maximum(ii, jj)
    # take unique non-diagonal non-negative elements
    take = np.intersect1d(
        np.unique(np.vstack([ii, jj]).T, axis=0, return_index=True)[1],
        np.where((ii < jj) & (ii >= 0))[0],
    )
    # make symmetric
    ii, jj = np.concatenate([ii[take], jj[take]]), np.concatenate(
        [jj[take], ii[take]]
    )
    xx = np.maximum(np.concatenate([xx[take], xx[take]]), min_dist)

    result = scipy.sparse.coo_matrix((xx, (ii, jj)), shape=(nnodes, nnodes))

    return result.tocsr()


def _write_gbbs_graph(dist, outfile):

    nnodes = dist.shape[0]
    ii, jj = dist.nonzero()
    xx = dist.data
    # sort by node indices
    o = np.lexsort([jj, ii])
    edges = jj[o]
    weights = xx[o] 
    # determine node offsets
    offsets = np.concatenate([[0], np.cumsum(np.bincount(ii[o]))[:-1]])
    nedges = len(xx)

    with open(outfile, "w") as outf:
        outf.write("WeightedAdjacencyGraph\n{0}\n{1}\n".format(nnodes, nedges))
        outf.write("\n".join("{0}".format(o) for o in offsets) + "\n")
        outf.write("\n".join("{0}".format(e) for e in edges) + "\n")
        outf.write("\n".join("{0:.10g}".format(w) for w in weights) + "\n")


def _convert_gbbs_linkage(Ls, max_dist=None):

    import scipy.cluster.hierarchy

    # find nodes with only one child
    num_children = Ls["parent"].value_counts()
    nonbinary = num_children.index[num_children == 1]
    if len(nonbinary) > 1:
        raise Exception(
            "convert_gbbs_linkage: not sure what to do if there is more than one node with only one child!"
        )
    elif len(nonbinary) == 1:
        for nb in nonbinary:
            child = Ls.index[Ls["parent"] == nb]
            if nb in Ls.index:
                # change parent node of child
                Ls.loc[child, "parent"] = Ls.loc[nb, "parent"]
                # remove this node
                Ls = Ls.drop(nb, axis=0)
            else:
                # if nb has no parent, drop child
                Ls = Ls.drop(child, axis=0)
        # adjust node indices
        ii = np.array(Ls.index)
        pp = np.array(Ls["parent"])
        ii[ii >= nb] = ii[ii >= nb] - len(nonbinary)
        pp[pp >= nb] = pp[pp >= nb] - len(nonbinary)
        Ls.index = ii
        Ls["parent"] = pp
    # sort by parent node index and by distance
    o = np.lexsort([Ls["wgh"].values, Ls["parent"].values])
    zz = []
    i = 0
    while i < len(o) - 1:
        if Ls["parent"][o[i]] == Ls["parent"][o[i + 1]]:
            if max_dist is not None:
                zz.append([o[i] + 1, o[i + 1] + 1, min(Ls["wgh"][o[i]], max_dist)])
            else:
                zz.append([o[i] + 1, o[i + 1] + 1, Ls["wgh"][o[i]]])
            i += 2
        else:
            # this should not happen
            i += 1
            raise Exception(
                "convert_gbbs_linkage: nodes with single children left in input array!"
            )
    return scipy.cluster.hierarchy.from_mlab_linkage(np.array(zz))


def _ts():
    return time.ctime(time.time())


def linkage (X, 
             dist=None,
             metric='euclidean', 
             metric_kwds={},
             method='single',
             n_neighbors=10,
             n_backup=10,
             random_state=0,
             n_trees=None,
             n_iters=None,
             max_candidates=60,
             low_memory=False,
             n_jobs=1,
             verbose=False,
             compressed=False,
             return_distance=False,
             use_pybindings=True):

    """Hierarchical agglomerative clustering using NNDescent for fast approximate 
    distance matrix calculation and gbbs for clustering. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    graph_data in many cases. 

    Parameters
    ----------
    data: (sparse) array of shape (n_samples, n_features)

    dist: alternatively, sparse distance matrix of shape (n_samples, n_samples)

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
            * hellinger
            * wasserstein-1d
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    method: string (optional, default='single')
        specify the the clustering scheme to determine the distance from a new node
        to the other nodes.at the moment supported are "single", "complete" and "average".

    n_neighbors: int (optional, default=10)
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    n_backup: int (optional, default=10)
        The number of non-nearest-neighbor backup datapoints to keep around.

    n_trees: int (optional, default=None)
        This implementation uses random projection forests for initializing the index
        build process. This parameter controls the number of trees in that forest. A
        larger number will result in more accurate neighbor computation at the cost
        of performance. The default of None means a value will be chosen based on the
        size of the graph_data.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    low_memory: boolean (optional, default=False)
        Whether to use a lower memory, but more computationally expensive
        approach to index construction.

    max_candidates: int (optional, default=None)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    n_iters: int (optional, default=None)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing. The default of None means
        a value will be chosen based on the size of the graph_data.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    compressed: bool (optional, default=False)
        Whether to prune out data not needed for searching the index. This will
        result in a significantly smaller index, particularly useful for saving,
        but will remove information that might otherwise be useful.

    use_pybindings: bool (optional, default=True)
        use gbbs python bindings, otherwise export distance matrix to filesystem
        and call gbbs directly (does not really work at the moment)

    return_distance: bool (optional, default=False)
        return approximate (sparse) distance matrix along with linkage

    verbose: bool (optional, default=False)
    """

    import os
    import sys
    import tempfile
    import numpy as np
    import pandas as pd
    from nndescent import NNDescent

    gbbs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'gbbs') 

    if dist is None:

        if verbose:
            print(_ts(), "running nndescent with {0} metric".format(metric))

        if not n_trees:
            n_trees = min(64,5+int(round(X.shape[0]**.5 / 20)))

        if not n_iters:
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

        si = NNDescent(X, 
                       n_neighbors=n_neighbors, 
                       n_backup=n_backup, 
                       metric=metric, 
                       metric_kwds=metric_kwds, 
                       random_state=0, 
                       n_trees=n_trees,
                       n_iters=n_iters,
                       max_candidates=max_candidates,
                       low_memory=low_memory,
                       n_jobs=n_jobs,
                       verbose=verbose,
                       compressed=compressed)

        ni, nd, bi, bd = si.neighbor_graph

        knn_inds = np.hstack([ni, bi])
        knn_dists = np.hstack([nd, bd])

        dist = _get_sparse_matrix_from_knn(knn_inds, knn_dists)

    if method in ['single', 'complete', 'average'] and use_pybindings:

        sys.path.append(gbbs_dir)
        sys.path.append(os.path.join(gbbs_dir, 'bazel-bin/pybindings'))
        import gbbs

        if verbose:
            print(_ts(), "running gbbs HierarchicalAgglomerativeClustering with {0} linkage".format(method))

        nz = dist.nonzero()
        m = np.vstack((nz[0],nz[1],dist.data)).T
        G = gbbs.numpyFloatEdgeListToSymmetricWeightedGraph(np.ascontiguousarray(m))
        L = G.HierarchicalAgglomerativeClustering(method,False)
        Z = _convert_gbbs_linkage(pd.DataFrame(L), max_dist=np.max(dist.data))

    else:

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_file = os.path.join(tmpdir, "graph.txt")
            linkage_file = os.path.join(tmpdir, "linkage.txt")
            _write_gbbs_graph(dist, graph_file)
            
            if verbose:
                print(_ts(), "running gbbs HierarchicalAgglomerativeClustering with {0} linkage (via command-line)".format(method))
            os.system(
                gbbs_dir
                + "/bazel-bin/benchmarks/Clustering/SeqHAC/HACDissimilarity -s -of {0} -linkage {1} {2} > /dev/null".format(
                    linkage_file, method, graph_file
                    )
                )
            if not os.path.isfile(linkage_file):
                raise Exception("gbbs clustering failed - linkage file missing!")
            
            Ls = pd.read_csv(
                linkage_file,
                index_col=0,
                header=None,
                names=["parent", "wgh"],
                delim_whitespace=True,
                )
            
            Z = _convert_gbbs_linkage(Ls, max_dist=np.max(dist.data))

    if return_distance:
        return Z, dist
    else:
        return Z
