"""
Test DSD-training flow with 3-layer auto-encoder with MIT-BIH arrhythmias database.

"""


import numpy as np
import matplotlib.pyplot as plt
import evaluation
import utilise
import data_split
import good as gd
import record as rd


epoch = 5
version = 'DSD_3layer_AE_32features_test'
classes_num = len(rd.good5)


########################################################################################################################
# split data for hold-out or k-fold cross validation
########################################################################################################################
data_set = np.loadtxt(fname='./dataset/data_set_good5_raw_length_256.csv', delimiter=',', dtype='float32')
label_set = np.loadtxt(fname='./dataset/label_set_good5_raw_length_256.csv', delimiter=',', dtype='float32')
data_len = data_set.shape[-1]


########################################################################################################################
#                                                normalization data
########################################################################################################################
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
norm = StandardScaler()
norm = norm.fit(X=data_set)
data_set = norm.transform(X=data_set)

# hold-out method shuffle
data_set, label_set = data_split.shuffle(data_set=data_set, label_set=label_set)

########################################################################################################################
#                                         split data for hold-out or k-fold cross validation
########################################################################################################################
train_rate = 0.8  # 80/20
train_X, train_Y, test_X, test_Y = data_split._data_split_5(data_set=data_set, label_set=label_set, train_rate=train_rate)


###################################################################################
# pre-train Autoencoder
###################################################################################
from DSD import ae_DenseLayer_3layer, ae_ParseLlayer_3layer, ae_reDenseLayer_3layer
# import dae
# import cae
# import sae
# import vae
# import convae
# ae = vae.Autoencoder(data_set=train_X)
# ae = dae.Autoencoder(data_set=train_X)
# ae = autoencoder.Autoencoder(data_set=train_X_noise)
ae = ae_DenseLayer_3layer.Autoencoder(data_len=data_len)
para_dict = ae.train(train_x=train_X, epoch=epoch)
#TODO plot weight value distribution
#TODO utilise.write_nd_data_v1(para_dict['w_enc'], filename='_Dense_w_enc')
#TODO utilise.write_nd_data_v1(para_dict['w_dec'], filename='_Dense_w_dec')

# ------------ mask ------------#
# para_dict = {'w_enc_h1': w_enc_h1, 'w_enc_h2': w_enc_h2, 'w_enc_h3': w_enc_h3, 'b_enc_h1': b_enc_h1, 'b_enc_h2': b_enc_h2, 'b_enc_h3': b_enc_h3,
#              'w_dec_h1': w_dec_h1, 'w_dec_h2': w_dec_h2, 'w_dec_h3': w_dec_h3, 'b_dec_h1': b_dec_h1, 'b_dec_h2': b_dec_h2, 'b_dec_h3': b_dec_h3}
sparsity = 0.5
mask_dict = {}
mask_dict['mask_w_enc_h1'], mask_dict['mask_b_enc_h1'] = gd.mask_top_k(weights=para_dict['w_enc_h1'], bias=para_dict['b_enc_h1'], sparsity=sparsity)
mask_dict['mask_w_enc_h2'], mask_dict['mask_b_enc_h2'] = gd.mask_top_k(weights=para_dict['w_enc_h2'], bias=para_dict['b_enc_h2'], sparsity=sparsity)
mask_dict['mask_w_enc_h3'], mask_dict['mask_b_enc_h3'] = gd.mask_top_k(weights=para_dict['w_enc_h3'], bias=para_dict['b_enc_h3'], sparsity=sparsity)
mask_dict['mask_w_dec_h1'], mask_dict['mask_b_dec_h1'] = gd.mask_top_k(weights=para_dict['w_dec_h1'], bias=para_dict['b_dec_h1'], sparsity=sparsity)
mask_dict['mask_w_dec_h2'], mask_dict['mask_b_dec_h2'] = gd.mask_top_k(weights=para_dict['w_dec_h2'], bias=para_dict['b_dec_h2'], sparsity=sparsity)
mask_dict['mask_w_dec_h3'], mask_dict['mask_b_dec_h3'] = gd.mask_top_k(weights=para_dict['w_dec_h3'], bias=para_dict['b_dec_h3'], sparsity=sparsity)
# FIXME: lack of shark figure (weight value distribution of threshold screened.)
#TODO utilise.write_nd_data_v1(?? * mask_w_enc, filename='_Sparse_w_enc')
#TODO utilise.write_nd_data_v1(?? * mask_w_dec, filename='_Sparse_w_dec')

ae = ae_ParseLlayer_3layer.Autoencoder(data_len=data_len, para_dict=para_dict, mask_dict=mask_dict)
train_X, train_Y = data_split.shuffle(data_set=train_X, label_set=train_Y)
para_dict = ae.train(train_x=train_X, epoch=3)
#TODO utilise.write_nd_data_v1(??, filename='_ReDense_w_enc')
#TODO utilise.write_nd_data_v1(??, filename='_ReDense_w_dec')

# ------------ recover ------------#
# TODO recover operation
para_dict = gd.restore_v1(para_dict=para_dict, mask_dict=mask_dict)
train_X, train_Y = data_split.shuffle(data_set=train_X, label_set=train_Y)
ae = ae_reDenseLayer_3layer.Autoencoder(data_len=data_len, para_dict=para_dict)

train_feat, test_feat, para_dict = ae.train(train_x=train_X, test_x=test_X, epoch=epoch*2)
# TODO utilise.write_nd_data_v1(??, filename='_')
train_feat, test_feat = np.array(train_feat).squeeze(), np.array(test_feat).squeeze()

# train_feat, test_feat, train_weight = np.array(train_feat).squeeze(), np.array(test_feat).squeeze(), np.array(train_weight).squeeze()
print('----------------------------------------- finished autoencoder')
###################################################################################


###################################################################################
# train SVM
###################################################################################
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
title = ['TP', 'TN', 'FP', 'FN', 'precision', 'recall(sensitivity)', 'specificity', 'accuracy', 'F1']

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.forest import ForestClassifier
# Random Forest Classifier
# cls = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=18), n_estimators=400, learning_rate=0.2, random_state=0)
cls = RandomForestClassifier(max_depth=10, n_estimators=500, criterion='gini', class_weight='balanced')

# MLP classifier
# cls = MLPClassifier(max_iter=400)

# one-vs-one classifier, LinearSVC
# cls = OneVsOneClassifier(estimator=LinearSVC(penalty='l2', loss='squared_hinge', max_iter=10000, class_weight='balanced'))

cls.fit(train_feat, train_Y)

utilise.write_nd_data_v0(_input=np.array(cls.feature_importances_), filename='./results/features_importances_'+version)

test_prediction = list(cls.predict(test_feat))
y_pred = test_prediction; y_true = test_Y
TP, TN, FP, FN, precision, recall, specificity, accuracy, F1 = evaluation.evaluation(prediction=y_pred, true=y_true, classes=classes_num)
print(TP, TN, FP, FN, precision, recall, specificity, accuracy, F1)
test_rsult = np.concatenate((TP, TN, FP, FN, precision, recall, specificity, accuracy, F1), axis=-1)
utilise.write_nd_data(_input=test_rsult, filename='_M3_results_'+version, classes=classes_num, title=title)


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)
utilise.write_nd_data_v1(cnf_matrix, filename='_M3_confusion_metrix_'+version)
plt.figure()

good5 = ['Norm', 'PVC', 'Paced', 'LBBB', 'RBBB']
utilise.plot_confusion_matrix(cnf_matrix, normalize=True, classes=good5, title='Confusion Matrix - DSD with 3-layer AE')
