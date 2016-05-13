"""Aggarwal and Yu evolutionary's algorithm."""

import os
import inspect
import numpy as np

from math import log, floor, ceil
from subprocess import call


class EvOutSe(object):
    """Aggarwal and Yu evolutionary's algorithm."""

    def __init__(self, m=None, phi=None, k=None, s=None):
        """Construtor.

        parameters:

            * m: Population size for evolutionary algorithm.
            * phi: Number of equi-depth grid ranges to use in each dimension.
            * k: Subspace dimensionality to search for.

        example:
            import numpy as np
            from sklearn.ensemble import EvOutSe

            data = np.genfromtxt("ionosphere.data.txt", delimiter=',')
            X_train = data[:200, :-1]
            X_test = data[200:, :-1]

            y_train = data[:200, -1]
            y_test = data[200:, -1]

            estimator = EvOutSe()
            pred = estimator.fit_predict(X_train, y_train, X_test, y_test)

            print roc_auc_score(y_test, pred)

        Ref: http://charuaggarwal.net/outl.pdf.
        """
        self.m = m
        self.phi = phi
        self.k = k
        self.s = s

    def _p(self, e):
        if e == 1:
            return 'a'
        else:
            return 'n'

    def fit_predict(self, X_train, y_train, X_test, y_test):
        """Fit and predict.

        Fit f_* with (X_train, X_test).

        return: pred = f_*(X_test)

        remark: y_train and y_test should be useless but Elki requires it...
        """
        # Find good default parameters
        N = X_train.shape[0] + X_test.shape[0]
        if self.m is None:
            self.m = int(1.3 * N)
        if self.phi is None:
            self.phi = int(ceil(log(N, 10)))
        if self.s is None:
            self.s = -3
        if self.k is None:
            self.k = int(floor(log(N / (self.s * self.s) + 1, self.phi)))

        # Parse data to Elki's format
        X = np.vstack((X_train, X_test))
        pytr = np.array([self._p(e) for e in y_train])
        pyte = np.array([self._p(e) for e in y_test])
        y = np.vstack((np.reshape(pytr, (pytr.size, 1)),
                       np.reshape(pyte, (pyte.size, 1))))
        data = np.hstack((X, y))

        install_dir = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))  # script directory

        # Now we can call Elki
        np.savetxt(".data.csv", data, delimiter=',', fmt="%s")
        call(["java", "-jar", install_dir + "/elki-0.7.2-SNAPSHOT.jar",
              "KDDCLIApplication",
              "-algorithm", "outlier.subspace.AggarwalYuEvolutionary",
              "-dbc.in", ".data.csv",
              "-ay.seed", str(0),
              "-ay.m", str(self.m),
              "-ay.k", str(self.k),
              "-ay.phi", str(self.phi),
              "-out", ".res",
              # "-time"
              ])
        call(["rm", ".data.csv"])

        # Process the results
        f = open(".res/aggarwal-yu-outlier_order.txt")
        res = np.zeros(N)
        for line in f:
            spl = line.split(' ')
            i = int(spl[0].split('=')[1]) - 1
            res[i] = -float(spl[-1].split('=')[1])

        f.close()
        call(["rm", "-r", ".res"])
        return res[X_train.shape[0]:]
