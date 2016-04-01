# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

from math import sqrt
import numpy as np
from scipy.stats import scoreatpercentile

from .base import NeighborsBase
from .base import KNeighborsMixin
from .base import UnsupervisedMixin

from ..metrics import pairwise_distances
from ..exceptions import NotFittedError

from ..utils.validation import check_is_fitted
from ..utils import check_array
from ..utils import _get_n_jobs

__all__ = ["LocalOutlierFactor"]


class LocalOutlierFactor(NeighborsBase, KNeighborsMixin, UnsupervisedMixin):
    """Unsupervised Outlier Detection using Local Outlier Factor (LOF)

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the degree depends on how isolated the object is
    with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered as outliers.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    p: integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting it is used to define the
        threshold on the decision function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only :meth:`k_neighbors` and :meth:`kneighbors_graph` methods.
    """
    def __init__(self, n_neighbors=5, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1):
        self._init_params(n_neighbors=n_neighbors,
                          algorithm=algorithm,
                          leaf_size=leaf_size, metric=metric, p=p,
                          metric_params=metric_params, n_jobs=n_jobs)

        self.contamination = contamination

    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        """
        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5]")

        super(LocalOutlierFactor, self).fit(X)

        self._k_distance_value_fit_X_, self.neighbors_indices_fit_X_ = (
            self._k_distance(None))

        # Compute decision_function over training samples to define threshold_:
        self.decision_function_train_ = -self._local_outlier_factor()  # as bigger is better

        self.threshold_ = -scoreatpercentile(
            -self.decision_function_train_, 100. * (1. - self.contamination))

        # XXX May be optimized (if X is not None) by only computing it for
        # X_sub_samples = neighbors of neighbors of X ?
        return self

    def predict(self, X=None):
        """Predict Local Outlier Factor of X.

        The local outlier factor (LOF) of a sample captures its
        supposed `degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            wrt to the training samples. If None, makes prediction on the
            training data without considering them as their own neighbors.

        Returns
        -------
        is_inlier : array of shape (n_samples,)
            Returns 1 for anomalies/outliers and -1 for inliers.
        """
        check_is_fitted(self, ["threshold_", "decision_function_train_",
                               "_k_distance_value_fit_X_",
                               "neighbors_indices_fit_X_" ])

        if X is not None:
            X = check_array(X, accept_sparse='csr')
            is_inlier = np.ones(X.shape[0], dtype=int)
        else:
            is_inlier = np.ones(self._fit_X.shape[0], dtype=int)

        is_inlier[self.decision_function(X) <= self.threshold_] = -1

        return is_inlier

    def decision_function(self, X=None):
        """Opposite of the Local Outlier Factor of X (as bigger is better).

        The local outlier factor (LOF) of a sample captures its
        supposed `degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features), optional
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples. If None, makes prediction on the
            training data without considering them as their own neighbors.

        Returns
        -------
        lof_scores : array of shape (n_samples,)
            The Local Outlier Factor of each input samples. The lower,
            the more abnormal.
        """
        check_is_fitted(self, ["threshold_", "decision_function_train_",
                               "_k_distance_value_fit_X_",
                               "neighbors_indices_fit_X_" ])

        # if X is None, make prediction on X=X_train without considered each
        # query point as its own neighbor.
        if X is None:
            return self.decision_function_train_

        else:
            X = check_array(X, accept_sparse='csr')
            return -self._local_outlier_factor(X)  # as bigger is better

    def _k_distance(self, X=None):
        """
        Compute the k_distance and the neighborhood of query samples wrt
        training samples self._fit_X.
        If X=None, neighbors of each sample self._fit_X are returned.
        In this case, the query point is not considered its own neighbor.
        """
        distances, neighbors_indices = self.kneighbors(
            X=X, n_neighbors=self.n_neighbors)
        k_dist = distances[:, self.n_neighbors - 1]

        return k_dist, neighbors_indices

    def _local_reachability_density(self, X, neighbors_indices):
        """The local reachability density (LRD)

        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples to compute the LRD w.r.t. self._fit_X
            If None, compute the LRD of self._fit_X w.r.t. itself.
            (In this case samples are not considered in their own
            neighborhood)

        neighbors_indices : array-like of shape (n_samples, self.n_neighbors)
            neighbors indices of X (or of self._fit_X if X is None)
            Added as a parameter to avoid re-computing it with 
            self._k_distance(X)[1] when X is not None (if X is None, it is
            equal to self.neighbors_indices_fit_X_)

        Returns
        -------
        local_reachability_density : array, shape (n_samples,)
            The local reachability density of each sample.
        """
        if X is None:
            X = self._fit_X

        n_samples = X.shape[0]
        reach_dist_array = np.zeros((n_samples, self.n_neighbors))

        for j in range(n_samples):
            neighbors_number = -1
            dist_j = pairwise_distances(X[j,:],
                                    self._fit_X[neighbors_indices[j, :],:],
                                    metric=self.effective_metric_)[0]
            for cpt, i in enumerate(neighbors_indices[j, :]):
                neighbors_number += 1
                reach_dist_array[j, neighbors_number] = np.max(
                    [self._k_distance_value_fit_X_[i], dist_j[cpt]])

        return 1. / np.mean(reach_dist_array, axis=1)

    def _local_outlier_factor(self, X=None):
        """Compute local outlier factor (LOF)

        The local outlier factor (LOF) of a sample captures its
        supposed `degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The points to compute the LOF w.r.t. training samples self._fit_X.
            Note that samples X are not considered in the neighborhood of
            self._fit_X for computing their local_reachability_density, but a
            sample both in X and self._fit_X is counted twice (thus considered
            in his own neighborhood).
            If None, compute LOF of self._fit_X w.r.t. to itself. (In this case
            samples are not counted twice, ie not considered in their own
            neighbourhood).

        Returns
        -------
        lof : array-like of shape (n_samples,)
            The LOF of X. The lower, the more normal.
        """
        if X is None:
            n_samples = self._fit_X.shape[0]
            neighbors_indices_X = self.neighbors_indices_fit_X_
        else:
            n_samples =  X.shape[0]
            neighbors_indices_X = self._k_distance(X)[1]
            
        # Compute the local_reachibility_density of samples X:
        X_lrd = self._local_reachability_density(X, neighbors_indices_X)
        lrd_ratios_array = np.zeros((n_samples, self.n_neighbors))

        # Avoid re-computing X_lrd if X is None:
        lrd = X_lrd if X is None else self._local_reachability_density(None, self.neighbors_indices_fit_X_)

        for j in range(n_samples):
            neighbors_number = -1
            for i in neighbors_indices_X[j, :]:
                neighbors_number += 1
                lrd_ratios_array[j, neighbors_number] = lrd[i] / X_lrd[j]

        return np.mean(lrd_ratios_array, axis=1)
