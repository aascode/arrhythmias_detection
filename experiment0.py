"""
baseline method: Wavelet + PCA
"""
import numpy as np
import matplotlib.pyplot as plt

import wfdb

import wavelet; wav = wavelet.Wavelet()
import evaluation
import utilise
import data_split
import record as rd
import random

cd_len = [134, 209, 255, 286, 310, 330, 348, 365]
classes_num = len(rd.good5)

wavelet_level = 8
data_set, label_set = [], []
for i in rd.all_48_record_list:
    record_name = rd.local_record_dir + str(i)
    record = wfdb.rdrecord(record_name=record_name, channels=[0], physical=False)
    ann = wfdb.rdann(record_name=record_name, extension='atr', return_label_elements=['symbol'])

    ids = np.in1d(ann.symbol, rd.good5)             ### These are what we need
    beats = np.array(ann.sample)[ids]           ### get rid of unnessary data
    label = np.array(ann.symbol)[ids]           ### only good[] annotation
    label = utilise.convert_label_5(_input=label, classes=rd.good5)
    sig = record.d_signal.ravel()

    for j in range(1, len(label) - 1):
        _from, _to = beats[j] - 90, beats[j] + 162  # data_len = 252
        if _from < 0:
            continue
        else:
            _, cd, _, _ = wav.wavelet_decompose(_input=sig[_from: _to], level=wavelet_level)
            data_set = np.concatenate((data_set, [cd])) if data_set != list([]) else [cd]
            label_set = np.concatenate((label_set, [label[j]])) if label_set != list([]) else [label[j]]
data_set = np.loadtxt(fname='data_set_good5.csv', delimiter=',', dtype='float32')
label_set = np.loadtxt(fname='label_set_good5.csv', delimiter=',', dtype='float32')
# utilise.write_nd_data_v1(_input=data_set, filename='data_set_good5')
# utilise.write_nd_data_v0(_input=label_set, filename='label_set_good5')
data_len = data_set.shape[-1]


########################################################################################################################
#                                                imbalanced process
########################################################################################################################
#TODO

########################################################################################################################
#                                                normalization data
########################################################################################################################
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
norm = Normalizer()
norm = norm.fit(X=data_set)
data_set = norm.transform(X=data_set)

# hold-out method shuffle
data_set, label_set = data_split.shuffle(data_set=data_set, label_set=label_set)


########################################################################################################################
#                                         split data for hold-out or k-fold cross validation
########################################################################################################################
train_rate = 0.8  # 80/20
#TODO fix this method to fit the class
train_X, train_Y, test_X, test_Y = data_split._data_split_5(data_set=data_set, label_set=label_set, train_rate=train_rate)


########################################################################################################################
#                                                      PCA
########################################################################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=12)
for j in cd_len:
    train_feat = pca.fit_transform(X=train_X[:, :j])
    if np.sum(pca.explained_variance_ratio_) >= 0.99:
        test_feat = pca.fit_transform(X=test_X[:, :j])
        break


########################################################################################################################
#                                              standardize features
########################################################################################################################
scaler = StandardScaler()
train_feat = scaler.fit_transform(X=train_feat)
test_feat = scaler.fit_transform(X=test_feat)


########################################################################################################################
#                                                  classifier
########################################################################################################################
random.seed(0)
title = ['TP', 'TN', 'FP', 'FN', 'precision', 'recall(sensitivity)', 'specificity', 'accuracy', 'F1']
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.forest import ForestClassifier

# one-vs-one classifier, LinearSVC
# cls = OneVsOneClassifier(estimator=LinearSVC(penalty='l2', loss='squared_hinge', max_iter=10000, class_weight='balanced'))

# one-vs-one classifier, SVC
# cls = OneVsOneClassifier(estimator=SVC(kernel='rbf', decision_function_shape='ovo', max_iter=50000, class_weight='balanced'))

# MLP classifier
# cls = MLPClassifier(max_iter=400)

# Random Forest Classifier
cls = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=20)
# cls = GradientBoostingClassifier(max_depth=6, learning_rate=0.2, subsample=0.8)



# cls = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6), n_estimators=600, learning_rate=1)
# cls = AdaBoostClassifier(base_estimator=GradientBoostingClassifier(max_depth=6, learning_rate=0.2, subsample=0.8), n_estimators=600)

# one-vs-rest classifier
# cls = OneVsRestClassifier(estimator=LinearSVC(loss='squared_hinge', max_iter=10000, class_weight='balanced'))
# cls = OneVsRestClassifier(estimator=LinearSVC(loss='l1', max_iter=10000, class_weight='balanced'))
cls.fit(train_feat, train_Y)


utilise.write_nd_data_v1(_input=np.array(cls.feature_importances_), filename='features_scores')


train_prediction = list(cls.predict(train_feat))
y_pred = train_prediction; y_true = train_Y
TP, TN, FP, FN, precision, recall, specificity, accuracy, F1 = evaluation.evaluation(prediction=y_pred, true=y_true, classes=classes_num)
print(TP, TN, FP, FN, precision, recall, specificity, accuracy, F1)
train_rsult = np.concatenate((TP, TN, FP, FN, precision, recall, specificity, accuracy, F1), axis=-1)
utilise.write_nd_data(_input=train_rsult, filename='train_result', classes=classes_num, title=title)

test_prediction = list(cls.predict(test_feat))
y_pred = test_prediction; y_true = test_Y
TP, TN, FP, FN, precision, recall, specificity, accuracy, F1 = evaluation.evaluation(prediction=y_pred, true=y_true, classes=classes_num)
print(TP, TN, FP, FN, precision, recall, specificity, accuracy, F1)
title = ['TP', 'TN', 'FP', 'FN', 'precision', 'recall(sensitivity)', 'specificity', 'accuracy', 'F1']
test_rsult = np.concatenate((TP, TN, FP, FN, precision, recall, specificity, accuracy, F1), axis=-1)
utilise.write_nd_data(_input=test_rsult, filename='test_result', classes=classes_num, title=title)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)
plt.figure()
utilise.plot_confusion_matrix(cnf_matrix, classes=rd.good5, title='Confusion matrix')
# print(cnf_matrix)
plt.savefig('confusion_matrix')
