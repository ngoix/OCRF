"""
==========================================
OneClassRF benchmark
==========================================

A test of OneClassRF on classical anomaly detection datasets.

"""
print(__doc__)
import pdb
from time import time
import numpy as np
# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.ensemble import OneClassRF
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.datasets import one_class_data
from sklearn.preprocessing import scale
from sklearn import grid_search  # let it for cv
from scipy.interpolate import interp1d

from sklearn.utils import shuffle as sh

np.random.seed(1)

nb_exp = 5

# TODO: find good default parameters for every datasets
# TODO: make an average of ROC curves over 20 experiments
# TODO: idem in bench_lof, bench_isolation_forest (to be launch from master)
#       bench_ocsvm (to be created), bench_ocrf (to be created)

# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'internet_ads', 'adult']

# continuous datasets:
datasets = ['http', 'smtp', 'shuttle', 'forestcover',
            'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt', 'adult']
# new: ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
#       'pima', 'wilt', 'adult']

# datasets = ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']
# datasets = ['wilt']
plt.figure(figsize=(25, 17))

for dat in datasets:
    # loading and vectorization
    X, y = one_class_data(dat)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    # # ###### XXX to be remove ################
    # # remove useless features:
    # useless_features = (X.max(axis=0) - X.min(axis=0)) == 0
    # for j in reversed(range(n_features)):
    #     if useless_features[j]:
    #         X = np.delete(X, (j), axis=1)
    # #########################################

    n_axis = 1000
    x_axis = np.linspace(0, 1, n_axis)
    tpr = np.zeros(n_axis)
    precision = np.zeros(n_axis)
    fit_time = 0
    predict_time = 0

    for ne in range(nb_exp):
        print 'exp num:', ne
        X, y = sh(X, y)
        # indices = np.arange(X.shape[0])
        # np.random.shuffle(indices)  # shuffle the dataset
        # X = X[indices]
        # y = y[indices]

        X_train = X[:n_samples_train, :]
        X_test = X[n_samples_train:, :]
        y_train = y[:n_samples_train]
        y_test = y[n_samples_train:]

        print('OneClassRF processing...')

        # # ############# cross-val: ##################################
        # # too much parameters: yields segmentation error
        # # parameters = {'max_depth': ['auto', 5, 10, 100, 1000],
        # #               'max_samples': [.01, .05, .1, .2, .3, 'auto'],
        # #               'max_features': [2, 5, 8, 10, 15],
        # #               'n_estimators': [5, 10, 20, 50]}
        # parameters = {'max_depth':  ['auto'],
        #               'max_samples': [.05, .1, 'auto'],
        #               'max_features': [max(int(0.5 * n_features),3), max(int(0.2 * n_features),3)],
        #               'max_features2': [min(10, n_features), min(5, n_features), 3],
        #               'n_estimators': [50, 100]}
        # model = OneClassRF()
        # clf = grid_search.GridSearchCV(model, parameters, refit=False, cv=2)
        # clf.fit(X_train, y_train)
        # print 'clf.best_params_', clf.best_params_
        # model.set_params(**clf.best_params_)
        # #############################################################

    # ############# bench results: ##########
    # http: {'max_features': 3, 'max_samples': 'auto', 'n_estimators': 50,
    #        'max_depth': 5} -> AUC 0.998 (alpha=1), 0.998 (alpha=0.1)
    # smtp: {'max_features': 3, 'max_samples': 0.1, 'n_estimators': 20,
    #        'max_depth': 100} -> AUC 0.981 (alpha=1), 0.98 (alpha=0.1)
    # forestcvr: {'max_features': 3, 'max_samples': 'auto', 'n_estimators': 50,
    #               'max_depth': 100} -> AUC 0.979 (alpha=1.)
    # shuttle: {'max_features': 3, 'max_samples': 0.1, 'n_estimators': 20,
    #           'max_depth': 5} -> AUC 1.000  (segm error when large max_depth)
    ###################

        # essayer avec max_features = max(int(0.2 * n_features), 3)
        # comment the following line if CV is performed above

        # test3 en faisant la vraie moyenne:
        # model = OneClassRF(max_depth='auto', max_samples=.1,
        #                    max_features=min(max(int(0.2 * n_features), 3),15),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # test4 (scores scaled):
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # test5 sans volume (scores scaled):
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # test6 score volume->1/depth (scores scaled): (reessayer en variant la profondeur) (pas de chgmt sans scale)
        # model = OneClassRF(max_depth=4,
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test7 juste la depth:
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test8 volume-> 1/2^depth: (un peu nul)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test9 idem mais en ajoutant _average_path_length: (nul)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test10 idem avec normalization (un peu nul) (effacé)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test10 idem mais sans moyenner chaque arbre: (un peu nul)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test11 idem (sans moyenner chaque arbre) avec 2^ (score intelligent): (idem, un peu nul)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.2 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # test14 (seulmt depth comme iforest)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.5 * n_features), 3),
        #                    max_features2=1.,
        #                    n_estimators=100)
        # # test15 (seulmt depth comme iforest)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=max(int(0.5 * n_features), 3),
        #                    max_features2=min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # # test16 (seulmt depth comme iforest) (stylé) sans + _average
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features2=1.,#min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # # test17 (seulmt depth comme iforest) (stylé)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features2=min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # test18 (seulmt depth comme iforest) avec random (et non best) (un peu moins stylé)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features2=min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # # test19 idem avec best avec + _average
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features_tree=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features_node=1.,#min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # # test20 idem avec max_features2='auto' (un tout petit peu moins bien)
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features_tree=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features_node='auto',#min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # # test19unsupervised idem test19 but unsupervised
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features_tree=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features_node=min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # # test19_1 idem test19 but max_features_node=1.
        # model = OneClassRF(max_depth='auto',
        #                    max_samples=max(int(0.1 * n_samples_train), 100),
        #                    max_features_tree=min(max(int(0.5 * n_features), 5), n_features),
        #                    max_features_node=1., #min(max(int(0.5 * n_features), 3), 5),
        #                    n_estimators=100)
        # test19_1_auto idem test19_1
        model = OneClassRF(max_depth='auto',
                           max_samples='auto', #max(int(0.1 * n_samples_train), 100),
                           max_features_tree='auto', #min(max(int(0.5 * n_features), 5), n_features),
                           max_features_node=1., #min(max(int(0.5 * n_features), 3), 5),
                           n_estimators=100)

        # apres: comparer avec max_features_node=1.

        # training only on normal data: (not supported in cv)
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

        tstart = time()
        model.fit(X_train)
        fit_time += time() - tstart
        tstart = time()

        scoring = model.predict(X_test)  # the lower, the more normal
        # pdb.set_trace()
        scoring = scale(scoring)
        predict_time += time() - tstart
        fpr_, tpr_ = roc_curve(y_test, scoring)[:2]
        f = interp1d(fpr_, tpr_)
        tpr += f(x_axis)
        tpr[0] = 0.

        precision_, recall_ = precision_recall_curve(y_test, scoring)[:2]

        # cluster: old version of scipy -> interpol1d needs sorted x_input
        arg_sorted = recall_.argsort()
        recall_ = recall_[arg_sorted]
        precision_ = precision_[arg_sorted]

        f = interp1d(recall_, precision_)
        precision += f(x_axis)

    tpr /= float(nb_exp)
    fit_time /= float(nb_exp)
    predict_time /= float(nb_exp)
    AUC = auc(x_axis, tpr)
    precision /= float(nb_exp)
    precision[0] = 1.
    AUPR = auc(x_axis, precision)

    # to print the parameters in plot, we need their type (some are not float):
    # if model.max_depth == 'auto':
    #     type_md = ' %s '
    # else:
    #     type_md = ' %0.2f '

    # if model.max_samples == 'auto':
    #     type_ms = ' %s '
    # else:
    #     type_ms = ' %0.2f '

    # plt.subplot(121)
    # label='%s (area = %0.3f, train-time: %0.2fs, test-time: %0.2fs), (%0.2f,' + type_ms + ', %0.2f,' + type_md + ')'
    # plt.plot(x_axis, tpr, lw=1, label=label % (dat, AUC, fit_time, predict_time, model.max_features, model.max_samples, model.n_estimators, model.max_depth))

    plt.subplot(121)
    plt.plot(x_axis, tpr, lw=1, label='%s (area = %0.3f, train-time: %0.2fs, test-time: %0.2fs)' % (dat, AUC, fit_time, predict_time, ))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('Receiver operating characteristic for OneClassRF', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 15})

    plt.subplot(122)
    plt.plot(x_axis, precision, lw=1, label='%s (area = %0.3f)'
             % (dat, AUPR))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('Precision-Recall curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 15})

plt.savefig('bench_oneclassrf_roc_pr_supervised_factorized_test19_1')
