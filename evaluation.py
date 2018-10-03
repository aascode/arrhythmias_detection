import numpy as np


def evaluation(prediction, true, classes):
    TP, FP, TN, FN, class_size = np.zeros(shape=classes),\
                                 np.zeros(shape=classes),\
                                 np.zeros(shape=classes),\
                                 np.zeros(shape=classes),\
                                 np.zeros(shape=classes)

    prediction = np.array(prediction)
    true = np.array(true)
    for i in range(classes):
        pred_buffer, true_buffer = prediction.copy(), true.copy()

        pred_p_ids, true_p_ids = np.where(pred_buffer == i), np.where(true_buffer == i)
        pred_n_ids, true_n_ids = np.where(pred_buffer != i), np.where(true_buffer != i)

        pred_buffer[pred_p_ids], true_buffer[true_p_ids] = 1, 1
        pred_buffer[pred_n_ids], true_buffer[true_n_ids] = 0, 0

        TP[i] = np.shape(np.where((pred_buffer + true_buffer) == 2))[1]
        TN[i] = np.shape(np.where((pred_buffer + true_buffer) == 0))[1]
        FP[i] = len(pred_buffer[pred_buffer > true_buffer])
        FN[i] = len(true_buffer[pred_buffer < true_buffer])

    precision = np.true_divide(TP, (TP + FP))
    recall = np.true_divide(TP, (TP + FN))
    specificity = np.true_divide(TN, (TN + FP))
    accuracy = np.true_divide((TP + TN), (TP + TN + FP + FN))
    fpr = np.true_divide(FP, (TN + FP))
    F1 = 2 * np.true_divide((precision * recall), (precision + recall))

    return TP, TN, FP, FN, precision, recall, specificity, fpr, accuracy, F1
    pass



def _evaluation(prediction, test_label):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    positive = 0
    negative = 0
    test_size = len(test_label)
    for i in range(test_size):
        if test_label[i] == 1:
            positive += 1
        else:
            negative += 1

        if prediction[i] == test_label[i]:
            if prediction[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if prediction[i] == 1:
                fp += 1
            else:
                fn += 1
    if (tp != 0):
        accuracy = np.float16(tp + tn) / np.float16(test_size)
        precision = np.float16(tp) / np.float16(tp + fp)
        recall = np.float16(tp) / np.float16(tp + fn)
        F_measure = np.float16(2 * precision * recall / (precision + recall))
        return positive, negative, tp, tn, accuracy, precision, recall, F_measure
    else:
        return positive, negative, tp, tn, 0, 0, 0, 0
