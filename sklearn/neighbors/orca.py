
from ..utils import check_array, timeout, max_time

import subprocess as sub
import pandas as pd
import csv

import pdb

import numpy as np

__all__ = ["Orca"]


class Orca():
    """Orca is a data-driven, unsupervised anomaly detection algorithm
    that uses a distance-based approach.
    It uses a novel pruning rule that allows it to run in nearly linear time.
    Orca was co-developed by Stephen Bay of ISLE and Mark Schwabacher
    of NASA ARC.

    Parameters
    ----------

    no_param : no, parameter, is, available (default = always)
        No parameter is available for the moment (we may add one for weights).
    """
    def __init__(self):
        print "initialization (nothing)"

    @timeout(max_time)
    def fit_predict(self, Xtrain, Xtest):
        Xtrain = check_array(Xtrain)
        print "Xtrain checked"
        Xtest = check_array(Xtest)
        print "Xtest checked"

        print "write data in 'Xtrain' and 'Xtest'"

        # CREATION OF THE XTRAIN FILE FOR DPREP
        dfXtrain = pd.DataFrame(data=Xtrain[0:, 0:])
        dfXtrain.to_csv("orca_linux_bin_static/Xtrain",
                        header=False, index=False, index_label=False)

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
        dfXtest = pd.DataFrame(data=Xtest[0:, 0:])
        dfXtest.to_csv("orca_linux_bin_static/Xtest",
                       header=False, index=False, index_label=False)

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
        p = sub.Popen(['./orca_linux_bin_static/dprep',
                       'orca_linux_bin_static/Xtrain',
                       'orca_linux_bin_static/Fields',
                       'orca_linux_bin_static/Xtrain.bin'],stdout=sub.PIPE,stderr=sub.PIPE)
        output, errors = p.communicate()
        p.wait()
        # print output
        # print errors

        print "Calling Dprep for Xtest"
        p2 = sub.Popen(['./orca_linux_bin_static/dprep',
                        'orca_linux_bin_static/Xtest',
                        'orca_linux_bin_static/Fields',
                        'orca_linux_bin_static/Xtest.bin'],stdout=sub.PIPE,stderr=sub.PIPE)
        output2, errors2 = p2.communicate()
        p2.wait()
        # print output2
        # print errors2


        print "Calling ORCA"

        # ================
        #|| ORCA OPTIONS ||
        # ================
        # nbOutlier computed
        nbOutlier = int(Xtest.shape[0] / 8.) # maybe ./8 as in paper "ilation forest"
        nbOutlierOption = str(nbOutlier)
        # nb of nearest neighbors considered by orca
        nbNN = str(5)
        # ======================
        #|| EN OF ORCA OPTIONS ||
        # ======================

        p3 = sub.Popen(['./orca_linux_bin_static/orca',
                        'orca_linux_bin_static/Xtest.bin',
                        'orca_linux_bin_static/Xtrain.bin', 'weights', '-n',
                        nbOutlierOption, '-k', nbNN],stdout=sub.PIPE,stderr=sub.PIPE)
        output3, errors3 = p3.communicate()
        p3.wait()
        # print output3
        # print errors3


        # ==========
        #|| PARSER ||
        # ==========
        topOutlierString = "Top outliers:"
        firstLetterIndex = output3.find(topOutlierString)
        recordString = "Record:"
        scoreString = "Score:"
        stringOutliersORCA = output3[firstLetterIndex+len(topOutlierString):len(output3)-1]
        outlierRank = 1

        outliersIndexes = np.zeros(nbOutlier)
        outliersScores = np.zeros(nbOutlier)
        scoring = np.zeros(Xtest.shape[0])

            #PARSING LOOP
        for i in range(nbOutlier-1):
            firstLetterIndex = stringOutliersORCA.find(recordString)
            firstLetterIndex = firstLetterIndex + len(recordString)
            lastLetterIndex = stringOutliersORCA.find(scoreString)
            # print "RECORD:"
            # print stringOutliersORCA[firstLetterIndex:lastLetterIndex]
            oind = int(stringOutliersORCA[firstLetterIndex:lastLetterIndex])
            outliersIndexes[i] = oind-1

            firstLetterIndex = lastLetterIndex + len(scoreString)
            outlierRank = outlierRank + 1
            outlierRankString = str(outlierRank) + ". "
            lastLetterIndex = stringOutliersORCA.find(outlierRankString)
            # print "SCORE:"
            # print stringOutliersORCA[firstLetterIndex:lastLetterIndex]
            osco = float(stringOutliersORCA[firstLetterIndex:lastLetterIndex])
            outliersScores[i] = osco

            stringOutliersORCA = stringOutliersORCA[lastLetterIndex+len(outlierRankString):len(output3)-1]
            # END OF PARSING LOOP

        # =================
        #|| END OF PARSER ||
        # =================

        scoring[list(outliersIndexes)] = outliersScores

        # MR PROPRE
        p4 = sub.Popen(['rm',
                        'orca_linux_bin_static/Xtrain',
                        'orca_linux_bin_static/Xtrain.bin',
                        'orca_linux_bin_static/Xtest',
                        'orca_linux_bin_static/Xtest.bin',
                        'orca_linux_bin_static/Fields',
                        'weights'], stdout=sub.PIPE, stderr=sub.PIPE)
        output4, errors4 = p4.communicate()
        p4.wait()
        print output4
        print errors4
        return scoring
