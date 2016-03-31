import pdb
from zipfile import ZipFile
from io import BytesIO, StringIO
import logging
from os.path import exists, join
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import numpy as np

from .base import get_data_home
from .base import Bunch
from .base import _pkl_filepath
from ..utils.fixes import makedirs
from ..externals import joblib
from ..utils import check_random_state




logger = logging.getLogger()


def fetch_spambase(data_home=None, download_if_missing=True,
                   random_state=None, shuffle=False):
    """Load the spambase dataset, downloading it if necessary.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest spambases with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest spambase dataset.

    """
    URL = ('http://archive.ics.uci.edu/ml/'
           'machine-learning-databases/spambase/spambase.zip')

    data_home = get_data_home(data_home=data_home)
    spambase_dir = join(data_home, "spambase")
    samples_path = _pkl_filepath(spambase_dir, "samples")
    targets_path = _pkl_filepath(spambase_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        makedirs(spambase_dir, exist_ok=True)
        logger.warning("Downloading %s" % URL)
        f = BytesIO(urlopen(URL).read())
        file_ = ZipFile(f, mode='r').open('spambase.data')
        Xy = np.genfromtxt(file_, delimiter=',')

        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    return Bunch(data=X, target=y, DESCR=__doc__)


def fetch_annthyroid(data_home=None, download_if_missing=True,
                     random_state=None, shuffle=False):
    """Load the annthyroid dataset, downloading it if necessary.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest annthyroids with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest annthyroid dataset.

    """
    URL1 = ('http://archive.ics.uci.edu/ml/'
            'machine-learning-databases/thyroid-disease/ann-train.data')
    URL2 = ('http://archive.ics.uci.edu/ml/'
            'machine-learning-databases/thyroid-disease/ann-test.data')

    data_home = get_data_home(data_home=data_home)
    annthyroid_dir = join(data_home, "annthyroid")
    samples_path = _pkl_filepath(annthyroid_dir, "samples")
    targets_path = _pkl_filepath(annthyroid_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        makedirs(annthyroid_dir, exist_ok=True)
        logger.warning("Downloading %s" % URL1)
        f = BytesIO(urlopen(URL1).read())
        # ou X = np.load(f)
        Xy1 = np.genfromtxt(f, delimiter=' ')

        logger.warning("Downloading %s" % URL2)
        f = BytesIO(urlopen(URL2).read())
        Xy2 = np.genfromtxt(f, delimiter=' ')
        Xy = np.r_[Xy1, Xy2]
        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    return Bunch(data=X, target=y, DESCR=__doc__)


def fetch_arrhythmia(data_home=None, download_if_missing=True,
                     random_state=None, shuffle=False):
    """Load the arrhythmia dataset, downloading it if necessary.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest arrhythmias with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest arrhythmia dataset.

    """
    URL = ('http://archive.ics.uci.edu/ml/'
           'machine-learning-databases/arrhythmia/arrhythmia.data')

    data_home = get_data_home(data_home=data_home)
    arrhythmia_dir = join(data_home, "arrhythmia")
    samples_path = _pkl_filepath(arrhythmia_dir, "samples")
    targets_path = _pkl_filepath(arrhythmia_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        makedirs(arrhythmia_dir, exist_ok=True)
        logger.warning("Downloading %s" % URL)
        f = BytesIO(urlopen(URL).read())
        # ou X = np.load(f)
        Xy = np.genfromtxt(f, delimiter=',')

        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    return Bunch(data=X, target=y, DESCR=__doc__)
