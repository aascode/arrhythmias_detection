import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

def plot_prc(y_true, y_scores, add, save_fig_name):
    average_precision = average_precision_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(str('Precision-Recall curve: AP={0:0.2f}' + add).format(average_precision))
    plt.savefig(save_fig_name)
    # plt.show()
    pass

#TODO: do six labels classification
def convert_label(_input, five_cls=True):
    """
    good = ['N', 'e', 'L', 'R', 'j',  -> class N
            'A', 'a', 'J', 'S',       -> class S
            'V', 'E',                 -> class V
            'F',                      -> class F
            '/', 'f', 'Q']            -> class Q
    """
    _l = len(_input)
    for i in range(_l):
        if _input[i] == 'N':   _input[i] = 0
        elif _input[i] == 'L': _input[i] = 0
        elif _input[i] == 'R': _input[i] = 0
        elif _input[i] == 'j': _input[i] = 0
        elif _input[i] == 'e': _input[i] = 0

        elif _input[i] == 'A': _input[i] = 1
        elif _input[i] == 'a': _input[i] = 1
        elif _input[i] == 'J': _input[i] = 1
        elif _input[i] == 'S': _input[i] = 1

        elif _input[i] == 'V': _input[i] = 2
        elif _input[i] == 'E': _input[i] = 2

        elif _input[i] == 'F': _input[i] = 3

        elif _input[i] == '/': _input[i] = 4
        elif _input[i] == 'f': _input[i] = 4
        elif _input[i] == 'Q': _input[i] = 4

        # elif (_input[i] == '/') & (five_cls == True): _input[i] = 4
        # elif (_input[i] == 'f') & (five_cls == True): _input[i] = 4
        # elif (_input[i] == 'Q') & (five_cls == True): _input[i] = 4
    return list(map(int, _input))
def convert_label_6(_input):
    """
    good6 = ['A', 'L', 'N', '/', 'R', 'V']
    N: normal
    A: atrial premature contraction
    V: premature ventricular contraction
    R: right bundle branch block
    L: legt bundle branch block
    /: paced beats
    """
    _l = len(_input)
    for i in range(_l):
        if _input[i] == 'A':   _input[i] = 0
        elif _input[i] == 'L': _input[i] = 1
        elif _input[i] == 'N': _input[i] = 2
        elif _input[i] == '/': _input[i] = 3
        elif _input[i] == 'R': _input[i] = 4
        elif _input[i] == 'V': _input[i] = 5
    return list(map(int, _input))

def convert_label_7(_input, classes):
    """
    good7 = ['N', 'V', '/', 'L', '!', 'R', 'E']
    N: normal
    V: premature ventricular contraction
    R: right bundle branch block
    L: left bundle branch block
    /: paced beats
    """
    _l = len(_input)
    for i in range(_l):
        if _input[i] ==   classes[0]: _input[i] = 0
        elif _input[i] == classes[1]: _input[i] = 1
        elif _input[i] == classes[2]: _input[i] = 2
        elif _input[i] == classes[3]: _input[i] = 3
        elif _input[i] == classes[4]: _input[i] = 4
        elif _input[i] == classes[5]: _input[i] = 5
        elif _input[i] == classes[6]: _input[i] = 6
    return list(map(int, _input))
def convert_label_5(_input, classes):
    """
    good7 = ['N', 'V', '/', 'L', 'R']
    N: normal
    V: premature ventricular contraction
    /: paced beats
    L: left bundle branch block
    R: right bundle branch block
    """
    _l = len(_input)
    for i in range(_l):
        if _input[i] ==   classes[0]: _input[i] = 0
        elif _input[i] == classes[1]: _input[i] = 1
        elif _input[i] == classes[2]: _input[i] = 2
        elif _input[i] == classes[3]: _input[i] = 3
        elif _input[i] == classes[4]: _input[i] = 4
    return list(map(int, _input))


def adding_noise(X, v):
    noise = v * np.random.rand(len(X[0]))
    return X + noise


def write_nd_data(_input, filename, classes, title):
    """
    write n-dimension data
    :param _input: input ndarray type
    :param filename: input string


    ----------------------
    example:

    """
    csvfile2 = open(str(filename) + '.csv', 'w', newline='')
    fieldnames, i = [], int(0)
    for i in range(0, len(_input)):
        if i % classes == 0:
            a = (i // classes)
            fieldnames.append(title[a])
        else:
            fieldnames.append(' ')
    writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
    writer.writeheader()

    writer = csv.writer(csvfile2)
    _input = np.reshape(_input, newshape=(1, len(_input)))
    for val in _input:
        writer.writerow(val)
    pass


def write_nd_data_v1(_input, filename):
    """
    write n-dimension data for data_set (n_sample, n_elements)
    :param _input: input ndarray type
    :param filename: input string


    ----------------------
    example:

    """
    csvfile2 = open(str(filename) + '.csv', 'w', newline='')

    writer = csv.writer(csvfile2)
    # _input = np.reshape(_input, newshape=(len(_input), 1))
    for val in _input:
        writer.writerow(val)
    csvfile2.close()
    pass

def write_nd_data_v0(_input, filename):
    """
    write n-dimension data for label_set
    :param _input: input ndarray type
    :param filename: input string


    ----------------------
    example:

    """
    csvfile2 = open(str(filename) + '.csv', 'w', newline='')

    writer = csv.writer(csvfile2)
    _input = np.reshape(_input, newshape=(len(_input), 1))
    for val in _input:
        writer.writerow(val)
    csvfile2.close()
    pass


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('_'+title)

