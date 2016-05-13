import numpy as np
import os
import inspect
import pyper as pr

from pandas import DataFrame


class RF:

    def __init__(self, ntree=100, nforests=25):
        self.ntree = ntree
        self.nforests = nforests

    def fit_predict(self, X_train, y_train, X_test, y_test):

        X = np.vstack((X_train, X_test))
        df = DataFrame(X)

        r = pr.R(use_pandas=True)
        r.assign("X", df)
        # r('print(X)')  # if I remove this line pyper get stuck.....

        install_dir = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))  # script directory
        r('source("' + install_dir + '/RF.R")')
        r('set.seed(0)')
        r('no.forests=' + str(int(self.nforests)))
        r('no.trees=' + str(int(self.ntree)))
        r('rfdist <- RFdist(X, mtry1=3, no.trees, no.forests, '
          'addcl1=T, addcl2=F, imp=T, oob.prox1=T)')
        r('labelRF=outlier(rfdist$cl1)')
        return -np.array(r.get('labelRF'))[X_train.shape[0]:]
