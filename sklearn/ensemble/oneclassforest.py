# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
# License: BSD 3 clause
from __future__ import division
import numbers
import numpy as np
from warnings import warn

from scipy.sparse import issparse

from ..externals import six
from ..tree import ExtraTreeClassifier
from ..utils import check_array

from .bagging import BaseBagging

from ..metrics import roc_auc_score

__all__ = ["OneClassRF"]


class OneClassRF(BaseBagging):
    """OneClass Forest Algorithm

    Return the anomaly score of each sample using the OneClassRF algorithm

    The OneClassRF 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of abnormality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.


    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=
                                        max(int(0.1 * n_samples_train), 100)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_features_tree : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        - If "auto", then `max_features_node=
                               min(max(int(0.5 * n_features), 5), n_features)`.
        - If int, then draw `max_features_tree` features.
        - If float, then draw `max_features_tree * X.shape[1]` features.

    max_features_node : int, float, string or None, optional (default=1.)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features_node` features at each split.
        - If float, then `max_features_node` is a percentage and
          `int(max_features_node * n_features)` features are considered at each
          split.
        - If "auto", then `max_features_node=sqrt(n_features)`.
        - If "sqrt", then `max_features_node=sqrt(n_features)`.
        - If "log2", then `max_features_node=log2(n_features)`.
        - If None, then `max_features_node=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features_node`` features.


    max_depth : int or float, optional (default="auto")
        The max_depth of the tree.
            - If int, then maximum depth is `max_depth`.
            - If float, then maximum depth is  `max_depth * max_samples`.
            - If "auto", then `max_depth = ceiling(log2 max_samples).

    scoring : string (default = 'depth')
        The scoring function used to make predictions.
            - If 'density': use local density averaged over leaves containing
              the observation to make prediction on.
            - If 'typical_cell_density': use local density of a
              typicall cell
            - If 'depth': use the same scoring function than Isolation Forest

    bootstrap : boolean, optional (default=False)
        Whether samples are drawn with replacement.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator
        If RandomState instance, random_state is the random number generator
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    """

    def __init__(self,
                 n_estimators=100,
                 max_samples='auto',
                 max_features_tree='auto',
                 max_features_node=1.,
                 max_depth='auto',
                 scoring='depth',
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(OneClassRF, self).__init__(
            base_estimator=ExtraTreeClassifier(
                max_features=max_features_node,
                criterion='oneclassgini',
                splitter='best',
                random_state=random_state),
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features_tree,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.max_depth = max_depth
        self.scoring = scoring
        self.max_features_tree = max_features_tree
        self.max_features_node = max_features_node

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficieny.

        Returns
        -------
        self : object
            Returns self.
        """
        # ensure_2d=False because there are actually unit test checking we fail
        # for 1d.
        X = check_array(X, accept_sparse=['csc'], ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # rnd = check_random_state(self.random_state)
        y = np.zeros(X.shape[0])  # rnd.uniform(size=X.shape[0])

        n_samples, n_features = X.shape

        # input rectangular cell kept in memory:
        self.lim_inf = X.min(axis=0)
        self.lim_sup = X.max(axis=0)

        # ensure that max_sample is in [1, n_samples]:
        if isinstance(self.max_samples, six.string_types):
            if self.max_samples == 'auto':
                max_samples = max(int(0.1 * n_samples), 100)
            else:
                raise ValueError('max_samples (%s) is not supported.'
                                 'Valid choices are: "auto", int or'
                                 'float' % self.max_samples)

        elif isinstance(self.max_samples, six.integer_types):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1]")
            max_samples = max(int(self.max_samples * X.shape[0]), 1)

        self.max_samples_ = max_samples

        # ############# for max_depth #############
        if isinstance(self.max_depth, six.string_types):
            if self.max_depth == 'auto':
                max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))
            else:
                raise ValueError('max_depth (%s) is not supported. '
                                 'Valid choices are: "auto", int or '
                                 'float' % self.max_depth)

        elif isinstance(self.max_depth, six.integer_types):
            # ensure that max_depth is in [1, max_samples]
            if self.max_depth > self.max_samples_:
                warn("max_depth (%s) is greater than "
                     "max_samples (%s). max_depth "
                     "will be set to max_samples for estimation."
                     % (self.max_depth, self.max_samples_))
                max_depth = self.max_samples_
            else:
                max_depth = self.max_depth
        else:  # float
            if not (0.0 < self.max_depth <= 1.0):
                raise ValueError("max_depth must be in (0, 1]")
            max_depth = int(self.max_depth * self.max_samples_)

        # ############# for max_features_tree #############
        if isinstance(self.max_features_tree, six.string_types):
            if self.max_features_tree == 'auto':
                max_features = min(max(int(0.5 * n_features), 5), n_features)
            elif self.max_features_tree == 'sqrt':
                max_features = int(np.sqrt(n_features))
            else:
                raise ValueError('max_features_tree (%s) is not supported. '
                                 'Valid choices are: "auto", int or '
                                 'float' % self.max_features_tree)
        elif isinstance(self.max_features_tree, (numbers.Integral, np.integer)):
            max_features = self.max_features_tree
        else:  # float
            max_features = int(self.max_features_tree * n_features)

        if not (0 < max_features <= n_features):
            raise ValueError("max_features_tree must be in (0, n_features]")
        ########################################
        super(OneClassRF, self)._fit(X, y, max_samples,
                                     max_depth=max_depth,
                                     max_features=max_features,
                                     sample_weight=sample_weight)
        return self

    def predict(self, X):
        """Predict anomaly score of X with the OneClassRF algorithm.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : array of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more normal.
        """
        # code structure from ForestClassifier/predict_proba
        # Check data
        X = check_array(X, accept_sparse='csr')
        n_samples = X.shape[0]

        n_samples_leaf = np.zeros((n_samples, self.n_estimators), order="f")
        if self.scoring == 'depth':
            depths = np.zeros((n_samples, self.n_estimators), order="f")
        if self.scoring == 'density':
            volume = np.zeros((n_samples, self.n_estimators), order="f")
            scores = np.zeros((n_samples, self.n_estimators), order="f")
        if self.scoring == 'typical_cell_density':
            volume = np.zeros((n_samples, self.n_estimators), order="f")

        for i, (tree, features) in enumerate(zip(self.estimators_,
                                                 self.estimators_features_)):
            leaves_index = tree.apply(X[:, features])
            node_indicator = tree.decision_path(X[:, features])

            n_samples_leaf[:, i] = tree.tree_.n_node_samples[leaves_index]
            if self.scoring == 'depth':
                depths[:, i] = np.asarray(node_indicator.sum(axis=1)).reshape(-1)
                # print depths[:, i]
            if self.scoring == 'density':
                volume[:, i] = tree.tree_.volume[leaves_index]
                scores[:, i] = np.divide(n_samples_leaf[:, i], volume[:, i])
                # scores[:, i] = n_samples_leaf[:, i] * (2 ** depths[:, i])
            if self.scoring == 'typical_cell_density':
                volume[:, i] = tree.tree_.volume[leaves_index]

        if self.scoring == 'depth':
            depths += _average_path_length(n_samples_leaf)
            scores_av = 2 ** (-depths.mean(axis=1) / _average_path_length(
                self.max_samples_))  # / n_samples_leaf.mean(axis=1)

        if self.scoring == 'density':
            scores_av = - scores.mean(axis=1)

        if self.scoring == 'typical_cell_density':
            scores_av = - n_samples_leaf.mean(axis=1) / volume.mean(axis=1)

        # scores_av = 2 ** (-depths.mean(axis=1) / _average_path_length(self.max_samples_))

        # depths += _average_path_length(n_samples_leaf)
        # depths /= _average_path_length(self.max_samples_)
        # for i in range(self.n_estimators):
        #     scores[:, i] = n_samples_leaf[:, i] * (2 ** depths[:, i])


        # one has to detect observation outside the input cell self.lim_inf/sup
        # (otherwise, can yields very normal score for them):
        out_index = (X > self.lim_sup).any(axis=1) + (X < self.lim_inf).any(
            axis=1)
        scores_av[out_index] = scores_av.max()
        return scores_av

    def decision_function(self, X):
        """Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : array, shape (n_samples,)
            The decision function of the input samples.

        """
        # minus as bigger is better (here less abnormal):
        return - self.predict(X)

    def score(self, X, y):  # return the auROC score
        """XXX missing docstring"""
        scoring = self.predict(X)
        return roc_auc_score(y, scoring)


def _average_path_length(n_samples_leaf):
    """ The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples, n_estimators), or int.
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : array, same shape as n_samples_leaf

    """
    if isinstance(n_samples_leaf, six.integer_types):
        if n_samples_leaf <= 1:
            return 1.
        else:
            return 2. * (np.log(n_samples_leaf) + 0.5772156649) - 2. * (
                n_samples_leaf - 1.) / n_samples_leaf

    else:

        n_samples_leaf_shape = n_samples_leaf.shape
        n_samples_leaf = n_samples_leaf.reshape((1, -1))
        average_path_length = np.zeros(n_samples_leaf.shape)

        mask = (n_samples_leaf <= 1)
        not_mask = np.logical_not(mask)

        average_path_length[mask] = 1.
        average_path_length[not_mask] = 2. * (
            np.log(n_samples_leaf[not_mask]) + 0.5772156649) - 2. * (
                n_samples_leaf[not_mask] - 1.) / n_samples_leaf[not_mask]

        return average_path_length.reshape(n_samples_leaf_shape)
