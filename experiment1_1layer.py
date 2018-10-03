"""
Test DSD-training flow with single-layer auto-encoder with MIT-BIH arrhythmias database.

"""


import numpy as np
import matplotlib.pyplot as plt
import evaluation
import utilise
import data_split
import good as gd
import record as rd


epoch = 5
version = 'DSD_1layer_AE_32features_1'
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
from DSD import ae_DenseLayer_1layer, ae_ParseLlayer_1layer, ae_reDenseLayer_1layer
# import dae
# import cae
# import sae
# import vae
# import convae
# ae = vae.Autoencoder(data_set=train_X)
# ae = dae.Autoencoder(data_set=train_X)
# ae = autoencoder.Autoencoder(data_set=train_X_noise)
ae = ae_DenseLayer_1layer.Autoencoder(data_len=data_len)
para_dict = ae.train(train_x=train_X, epoch=epoch)
utilise.write_nd_data_v1(para_dict['w_enc'], filename='./results/weight_values_Dense_encoder')
utilise.write_nd_data_v1(para_dict['w_dec'], filename='./results/weight_values_Dense_decoder')

# ------------ mask ------------#
sparsity = 0.5
mask_w, mask_b = gd.mask_top_k(weights=para_dict['w_enc'], bias=para_dict['b_enc'], filename='weight_values_Pruning_encoder')
mask_w_enc, mask_b_enc = mask_w, mask_b
# w_p_enc = w_enc * mask_w; w_r_enc = w_enc * ~mask_w
# b_p_enc = b_enc * mask_b; b_r_enc = b_enc * ~mask_b
mask_w, mask_b = gd.mask_top_k(weights=para_dict['w_dec'], bias=para_dict['b_dec'], filename='weight_values_Pruning_decoder')
mask_w_dec, mask_b_dec = mask_w, mask_b
# w_p_dec = w_dec * mask_w; w_r_dec = w_dec * ~mask_w
# b_p_dec = b_dec * mask_b; b_r_dec = b_dec * ~mask_b
# print(np.shape(w_p_enc), np.shape(b_p_enc), np.shape(w_p_dec), np.shape(b_p_dec))
mask_dict = {'mask_w_enc': mask_w_enc, 'mask_b_enc': mask_b_enc, 'mask_w_dec': mask_w_dec, 'mask_b_dec': mask_b_dec}

train_X, train_Y = data_split.shuffle(data_set=train_X, label_set=train_Y)
ae = ae_ParseLlayer_1layer.Autoencoder(data_len=data_len, para_dict=para_dict, mask_dict=mask_dict)
para_dict = ae.train(train_x=train_X, epoch=epoch)
utilise.write_nd_data_v1(para_dict['w_enc'], filename='./results/weight_values_Sparse_encoder')
utilise.write_nd_data_v1(para_dict['w_dec'], filename='./results/weight_values_Sparse_decoder')

# ------------ recover ------------#
# TODO recover operation
para_dict = gd.restore_v0(para_dict=para_dict, mask_dict=mask_dict)
train_X, train_Y = data_split.shuffle(data_set=train_X, label_set=train_Y)
ae = ae_reDenseLayer_1layer.Autoencoder(data_len=data_len, para_dict=para_dict)
train_feat, test_feat, para_dict = ae.train(train_x=train_X, test_x=test_X, epoch=epoch*3)
utilise.write_nd_data_v1(para_dict['w_enc'], filename='./results/weight_values_ReDense_encoder')
utilise.write_nd_data_v1(para_dict['w_dec'], filename='./results/weight_values_ReDense_decoder')
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
utilise.write_nd_data(_input=test_rsult, filename='_DSD_results_'+version, classes=classes_num, title=title)


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)
utilise.write_nd_data_v1(cnf_matrix, filename='_DSD_confusion_metrix_'+version)
plt.figure()

good5 = ['Norm', 'PVC', 'Paced', 'LBBB', 'RBBB']
utilise.plot_confusion_matrix(cnf_matrix, normalize=True, classes=good5, title='Confusion Matrix - DSD with 1-layer AE')
