# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

from ..utils import check_array

import subprocess as sub
import pandas as pd
import csv

__all__ = ["Orca"]


class Orca():
    """Orca is a data-driven, unsupervised anomaly detection algorithm 
    that uses a distance-based approach. 
    It uses a novel pruning rule that allows it to run in nearly linear time. 
    Orca was co-developed by Stephen Bay of ISLE and Mark Schwabacher of NASA ARC.

    Parameters
    ----------

    no_param : no, parameter, is, available (default = always)
        No parameter is available for the moment (we may add one for weights).
    """
    def __init__(self):
	   print "initialization (nothing)"

    def fit_predict(self, Xtrain, Xtest):
        Xtrain=check_array(Xtrain)
        print "Xtrain checked"
        Xtest=check_array(Xtest)
        print "Xtest checked"

        print "write data in 'Xtrain' and 'Xtest'"

        # CREATION OF THE XTRAIN FILE FOR DPREP
        dfXtrain = pd.DataFrame(data=Xtrain[0:,0:])
        dfXtrain.to_csv("orca_linux_bin_static/Xtrain", header=False, index=False, index_label=False)

        # IDEM EN PLUS LOURD
        # fileXtrain = open('Xtrain', 'w')
        # print "Xtrain opened"
        # print fileXtrain
        # for i in range(Xtrain.shape[0]):
        #     for j in range(Xtrain.shape[1]):
        #         fileXtrain.write(str(Xtrain[i,j]))
        #         if j==Xtrain.shape[1]-1:
        #             fileXtrain.write('\n')
        #         else:
        #             fileXtrain.write(', ')

        # CREATION OF THE XTEST FILE FOR DPREP 
        dfXtest = pd.DataFrame(data=Xtest[0:,0:])
        dfXtest.to_csv("orca_linux_bin_static/Xtest", header=False, index=False, index_label=False)

        # IDEM EN PLUS LOURD
        # fileXtest = open('Xtest', 'w')
        # print "Xtest opened"
        # print fileXtest

        # for i in range(Xtest.shape[0]):
        #     for j in range(Xtest.shape[1]):
        #         fileXtest.write(str(Xtest[i,j]))
        #         if j==Xtest.shape[1]-1:
        #             fileXtest.write('\n')
        #         else:
        #             fileXtest.write(', ')

        # CREATION OF THE FIELD FILE FOR DPREP
        fileFields = open('orca_linux_bin_static/Fields', 'w')
        print "Fields opened"
        print fileFields
        for j in range(Xtest.shape[1]):
            line = "feature" + str(j) + ": continuous.\n"
            fileFields.write(line)
        fileFields.close()

        print "Calling Dprep for Xtrain"
        p = sub.Popen(['./orca_linux_bin_static/dprep','orca_linux_bin_static/Xtrain','orca_linux_bin_static/Fields','orca_linux_bin_static/Xtrain.bin'],stdout=sub.PIPE,stderr=sub.PIPE)
        output, errors = p.communicate()
        p.wait()
        print output
        print errors

        print "Calling Dprep for Xtest"
        p2 = sub.Popen(['./orca_linux_bin_static/dprep','orca_linux_bin_static/Xtest','orca_linux_bin_static/Fields','orca_linux_bin_static/Xtest.bin'],stdout=sub.PIPE,stderr=sub.PIPE)
        output2, errors2 = p2.communicate()
        p2.wait()
        print output2
        print errors2

        print "Calling ORCA"
        p3 = sub.Popen(['./orca_linux_bin_static/orca','orca_linux_bin_static/Xtest.bin','orca_linux_bin_static/Xtrain.bin','weights'],stdout=sub.PIPE,stderr=sub.PIPE)
        #../data/adult.bin','../data/adult.bin','../data/adult.weights'],stdout=sub.PIPE,stderr=sub.PIPE)
        output3, errors3 = p3.communicate()
        print output3
        print errors3

        # MR PROPRE
        p4 = sub.Popen(['rm','orca_linux_bin_static/Xtrain','orca_linux_bin_static/Xtrain.bin','orca_linux_bin_static/Xtest','orca_linux_bin_static/Xtest.bin','orca_linux_bin_static/Fields','weights'],stdout=sub.PIPE,stderr=sub.PIPE)
        output4, errors4 = p4.communicate()
        print output4
        print errors4

	return self

        """Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        """

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
