import numbers
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state, check_array


# XXX : add tests


class OneClassForest(BaseEstimator):
    """One Class Forest Algorithm

    Return the anomaly score of each sample in X with the iForest algorithm

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training data

    n_estimators : int
        The number of trees to build

    max_samples : int or float, optional (default=256)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Attributes
    -------
    `forest_` : list
        The estimated forest.
    """
    def __init__(self, n_estimators=100, max_samples=256, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """XXX missing docstring"""
        psi = self.max_samples
        n_samples = X.shape[0]
        if (not isinstance(psi, (numbers.Integral, np.integer)) and
                (0.0 < psi <= 1.0)):
            psi = int(psi * n_samples)
        self.forest_ = _build_iforest(
            X, t=self.n_estimators, psi=psi, random_state=self.random_state)
        return self

    def predict(self, X):
        """XXX missing docstring"""
        return _iforest_score(X, self.forest_)

    def decision_function(self, X):
        """XXX missing docstring"""
        return -self.predict(X)  # minus as bigger is better (here less abnormal)

    def score(self, X, y):  # return the auROC score
        """XXX missing docstring"""
        scoring = self.predict(X)
        return roc_auc_score(y, scoring)


def _iforest_score(X, forest):
    """Return the anomaly score of each sample in X with the iForest algorithm

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to evaluate anomaly score.

    forest : list
        The forest.

    Returns
    -------
    scores : ndarray
        scores[i] is the anomaly score of the sample X[i, :]
    """
    n_samples = X.shape[0]
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        mean_path_length = 0
        x = X[i, :]
        for tree in forest:
            path_length = 0
            while isinstance(tree, _InNode):
                if tree.compare(x):
                    tree = tree.right
                else:
                    tree = tree.left
                path_length += 1
            path_length += _cost(tree.size)
            mean_path_length += path_length
        mean_path_length /= len(forest)
        scores[i] = np.power(2, - mean_path_length / _cost(n_samples))
    return scores


def _cost(n):
    """ The average path length in a n samples iTree, which is equal to
        the average path length of an unsuccessful BST search since the
        latter has the same structure as an isolation tree.
    """
    if n <= 1:
        return 0.
    else:
        return 2. * _harmonic(n) - 2. * (n - 1.) / n


def _harmonic(n):
    """Harmonic number
    """
    return np.log(n) + 0.5772156649


def _build_iforest(X, t, psi, random_state=None):
    """Construct a list of binary trees

    isolation forest set of isolation trees

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data used to build the trees.

    t : int
        The number of trees to build

    psi : int
        the sub-sampling size used to build a tree

    Returns
    -------
    forest : list
        List of binary trees
    """
    rng = check_random_state(random_state)
    n_samples = X.shape[0]
    max_level = int(np.ceil(np.log2(psi)))
    forest = []
    for _ in range(t):
        sample = rng.permutation(n_samples)[:psi]
        X_sample = X[sample, :]  # XXX copy should be avoided
        forest.append(
            _itree(X_sample, current_level=0, max_level=max_level, rng=rng))
    return forest


def _itree(X, current_level, max_level, rng, split_bool_all=None, volume=None, volume_init=None):
    """Construct a binary tree, named isolation tree.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data used to build the isolation tree.

    current_level : int
        The level of the root of this itree is located

    max_level : int
        the maximum level authorized

    rng : instance of numpy.RandomState
        Control random number generation. Container for the pseudo-random
        number generator.

    split_bool_all : samples in the node (if None: every samples are considered)

    volume : volume of the node

    volume_init : volume of the initial cube containing all the samples

    Returns
    -------
    tree : instance of inNode
        The computed tree.
    """
    n_initial, n_features = X.shape

    if volume is None:
        volume = np.prod(np.maximum(np.max(X, axis=0) - np.min(X, axis=0), 0.001 * np.ones(n_features)))

    if volume_init is None:
        volume_init = np.prod(np.maximum(np.max(X, axis=0) - np.min(X, axis=0), 0.001 * np.ones(n_features)))
        
    if split_bool_all is None:
        split_bool_all = np.ones((n_initial,), dtype=bool)
    n_samples = sum(split_bool_all)

    if current_level >= max_level or n_samples <= 1:
        tree = _ExNode(n_samples)
    else:
        # feature to split on :
        split_feature = rng.randint(0, n_features)
        x = X[split_bool_all, split_feature]
        min_split = np.min(x)
        max_split = np.max(x)

        # find optimal split_value:
        gini_one_class_index = 2

        x_sorted = np.sort(x)
        for i in range(len(x)):
            split_value_ = x_sorted[i]
            # split_value_ = min_split + (max_split - min_split) * rng.rand()

            split_bool = x < split_value_
            cpt = 0
            split_bool_all_left_ = np.copy(split_bool_all)
            split_bool_all_right_ = np.copy(split_bool_all)
            for i in range(n_initial):
                if split_bool_all[i] == True:
                    split_bool_all_left_[i] = split_bool[cpt]
                    split_bool_all_right_[i] = not(split_bool[cpt])
                    cpt += 1

            X_right = X[split_bool_all_right_]
            X_left = X[split_bool_all_left_]
            n_right = sum(split_bool_all_right_)
            n_left = sum(split_bool_all_left_)
            volume_left_ = volume * max((split_value_ - min_split), 0.001)
            volume_right_ = volume * max((max_split - split_value_), 0.001)
            # print n_right, n_samples, volume_right_, volume, volume_init
            gini_one_class_index_right = 0.5 * (n_right * n_samples * volume_right_ / volume_init) / (n_right + n_samples * volume_right_ / volume_init) ** 2
            gini_one_class_index_left = 0.5 * (n_left * n_samples * volume_left_ / volume_init) / (n_left + n_samples * volume_left_ / volume_init) ** 2
            gini_one_class_index_ = float(n_right) / n_samples * gini_one_class_index_right  + float(n_left) / n_samples  * gini_one_class_index_left
            if gini_one_class_index_ < gini_one_class_index:
                split_value = split_value_
                split_bool_all_right = split_bool_all_right_
                split_bool_all_left = split_bool_all_left_
                volume_right = volume_right_
                volume_left = volume_left_
                gini_one_class_index = gini_one_class_index_

        left = _itree(X, current_level + 1, max_level, rng, split_bool_all_left, volume=volume_right, volume_init=volume_init)
        right = _itree(X, current_level + 1, max_level, rng, split_bool_all_right, volume=volume_right, volume_init=volume_init)
        tree = _InNode(split_feature, split_value, left, right)
    return tree


class _InNode:
    """Represent an internal node of a binary tree

    It is by extension a binary tree as a whole.
    Has two children, self.left and self.right, of type 'instance'.

    Parameters
    ----------
    split_feature : int
        Spliting feature.

    split_value : float
        Spliting value on the dimension split_feature.

    left : instance of _InNode or _ExNode
        Left descendant of the actual node. Data whose splitting feature is
        less than the spliting value.

    right : instance of _InNode or _ExNode
        Right descendant of the actual node. Data whose splitting feature is
        greater than the spliting value.
    """
    def __init__(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right

    def compare(self, x):
        return x[self.split_feature] >= self.split_value


class _ExNode:
    """Represent an external node (leaf) of an isolation tree.

    The set of all external nodes forms a partition of the data space.
    Each sample belongs to one and only one exNode.

    Parameters
    ----------
    size : int
       Number of data contained in this node.
    """
    def __init__(self, size=1):
        self.size = size
