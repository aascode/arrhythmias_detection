import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import wfdb
import utilise
import record as rd
import wavelet; wav = wavelet.Wavelet(); wavelet_level = 4; use = 'wavelet'
import time
import logging
mitdb_fs = 360
_5_mins = mitdb_fs * 300

"""
For classify the SVEB(Superventricular Premature Beat) and VEB(Ventricular Ectopic Beat)
Total 48 ECG recordings, 23(from 100 to 214) and 25 (from 200 to 234, which are rare but clinically important).
    exclusive recordings of 102, 104, 107 and 217 which mainly exists paced beats.

N = "N", 'NORMAL', 'Normal beat'
    "L", 'LBBB', 'Left bundle branch block beat'
    "R", 'RBBB', 'Right bundle branch block beat'
    "j", 'NESC', 'Nodal (junctional) escape beat'

S = "a", 'ABERR', 'Aberrated atrial premature beat'
    "J", 'NPC', 'Nodal (junctional) premature beat'
    "S", 'SVPB', 'Premature or ectopic supraventricular beat'

V = "V", 'PVC', 'Premature ventricular contraction'
    "E", 'VESC', 'Ventricular escape beat'

F = "F", 'FUSION', 'Fusion of ventricular and normal beat'

Q = "/", 'PACE', 'Paced beat'
    "f", 'PFUS',  'Fusion of paced and normal beat'
    "Q", 'UNKNOWN', 'Unclassifiable beat'
"""

training_set = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
    '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
]

testing_set = [
    '100', '103', '105', '111', '113', '117', '121', '123', '124', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]

def load_data_test(_list=None):
    data_set = []
    label_set = []
    for i in _list:
        record_name = rd.local_record_dir + str(i)
        ann = wfdb.rdann(record_name=record_name, extension='atr', return_label_elements=['symbol'])

        ids = np.in1d(ann.symbol, rd.good6)    # These are what we need
        beats = np.array(ann.sample)[ids]  # get rid of unnessary data
        label = np.array(ann.symbol)[ids]  # only good[] annotation
        label = utilise.convert_label_6(label)

        for j in range(len(beats) - 1):
            _from, _to = beats[j] - 128, beats[j] + 128  # data_len = 256
            if _from < 0:
                continue
            label_set = np.concatenate((label_set, [label[j]]), axis=-1) if label_set != list([]) else [label[j]]
    return data_set, label_set


def load_data_v0():
    data_set = []
    label_set = []
    start = time.clock()
    for i in rd.all_44_record_list:
        record_name = rd.local_record_dir + str(i)
        record = wfdb.rdrecord(record_name=record_name, channels=[0], physical=False)
        ann = wfdb.rdann(record_name=record_name, extension='atr', return_label_elements=['symbol'])

        ids = np.in1d(ann.symbol, rd.good6)    # These are what we need
        beats = np.array(ann.sample)[ids]  # get rid of unnessary data
        label = np.array(ann.symbol)[ids]  # only good[] annotation
        label = utilise.convert_label_6(label)
        sig = record.d_signal.ravel()

        for j in range(len(beats) - 1):
            _from, _to = beats[j] - 128, beats[j] + 128  # data_len = 256
            if _from < 0:
                continue
            else:
                if use == 'wavelet':
                    ca, cd, a, d = wav.wavelet_decompose(_input=sig[_from: _to], level=wavelet_level)
                    buffer = cd
                elif use == 'raw':
                    buffer = sig[_from: _to]
            data_set = np.concatenate((data_set, [buffer])) if data_set != list([]) else [buffer]
            label_set = np.concatenate((label_set, [label[j]]), axis=-1) if label_set != list([]) else [label[j]]
        del buffer

    end = time.clock()
    print(str(start+end), data_set.shape, label_set.shape)
    return data_set, label_set


def load_data_v1():
    """
    add rri
    """
    data_set = []
    label_set = []
    rr_set = []
    start = time.clock()
    for i in rd.all_44_record_list:
        record_name = rd.local_record_dir + str(i)
        record = wfdb.rdrecord(record_name=record_name, channels=[0], physical=False)
        ann = wfdb.rdann(record_name=record_name, extension='atr', return_label_elements=['symbol'])

        ids = np.in1d(ann.symbol, rd.good6)    # These are what we need
        beats = np.array(ann.sample)[ids]  # get rid of unnessary data
        label = np.array(ann.symbol)[ids]  # only good[] annotation
        label = utilise.convert_label_6(label)
        sig = record.d_signal.ravel()

        for j in range(len(beats) - 1):
            _from, _to = beats[j] - 128, beats[j] + 128  # data_len = 256
            if _from < 0:
                continue
            else:
                if use == 'wavelet':
                    ca, cd, a, d = wav.wavelet_decompose(_input=sig[_from: _to], level=wavelet_level)
                    buffer = cd
                elif use == 'raw':
                    buffer = sig[_from: _to]
                    pre_rr = beats[j] - beats[j - 1]; post_rr = beats[j + 1] - beats[j]; dif_rr = post_rr - pre_rr
                    buffer_rr = np.concatenate(([pre_rr], [post_rr], [dif_rr]), axis=-1)
            data_set = np.concatenate((data_set, [buffer])) if data_set != list([]) else [buffer]
            label_set = np.concatenate((label_set, [label[j]]), axis=-1) if label_set != list([]) else [label[j]]
            rr_set = np.concatenate((rr_set, [buffer_rr])) if rr_set != list([]) else [buffer_rr]
    del buffer, buffer_rr

    end = time.clock()
    print(str(start+end), data_set.shape, label_set.shape)
    return data_set, label_set



#TODO: the training set of Huang Jiao
def make_Huang_data_set():
    pass


############################################################
def read_data(list, sampfrom=0, sampto=None):
    data_set, label_set = {}, {}
    for i in list:
        record_name = rd.local_record_dir + str(i)
        print('record_name: ', i)
        sig, label, beats_index = read(record_name=record_name, sampfrom=sampfrom, sampto=sampto)

        _d, _l = windowed_with_feats(_sig=sig, _label=label, _index=beats_index)
        data_set[i], label_set[i] = _d, _l
    return data_set, label_set


def read(record_name, sampfrom=0, sampto=None):
    record = wfdb.rdrecord(record_name=record_name, channels=[0], physical=False, sampfrom=sampfrom, sampto=sampto)
    sig = record.d_signal.ravel()

    if sampfrom > 0:
        ann = wfdb.rdann(record_name=record_name, extension='atr', sampto=sampfrom)
        _good_index = np.in1d(ann.symbol, rd.good5)
        _len = np.shape(_good_index)[0]
        ann = wfdb.rdann(record_name=record_name, extension='atr', sampto=sampto)
        good_index = np.in1d(ann.symbol, rd.good5)
        good_index = good_index[_len:]

        label = np.array(ann.symbol[_len:])[good_index]
        label = utilise.convert_label(_input=label)

        beats_index = np.array(ann.sample[_len:])[good_index] - _5_mins
    else:
        ann = wfdb.rdann(record_name=record_name, extension='atr', sampto=sampto)
        good_index = np.in1d(ann.symbol, rd.good5)

        label = np.array(ann.symbol)[good_index]
        label = utilise.convert_label(_input=label)

        beats_index = np.array(ann.sample)[good_index]
    return sig, label, beats_index


def windowed_with_feats(_sig, _label, _index):
    _d, _l = [], []
    for i in range(1, len(_label) - 1):        # last one no count
        _from, _to = _index[i] - 128, _index[i] + 128
        if len(_sig[_from: _to]) == 0: print('here')
        if _from < 0: continue
        else:
        # elif len(_sig[_from: _to]) == 256:
            # raw: 256
            raw = _sig[_from: _to]

            # rr features: 3
            pre_rr = _index[i] - _index[i - 1]
            post_rr = _index[i + 1] - _index[i]
            d_rr = post_rr - pre_rr

            # basic features: Max, min, Mean, Variation, Standard Deviation, Median: 3
            Vari = [np.var(_sig[_from: _to])]
            Std = [np.std(_sig[_from: _to])]
            Medi = [np.median(_sig[_from: _to])]

            # wavelet features: 260
            _, cd, _, _ = wav.wavelet_decompose(_input=_sig[_from: _to], level=wavelet_level)

            # raw data: 522
            buffer = np.concatenate((raw, [pre_rr], [post_rr], [d_rr], Vari, Std, Medi, cd), axis=-1)
            pass

        if len(buffer) == 525:
            _d = np.concatenate(((_d, [buffer]))) if _d != list([]) else [buffer]
            _l = np.concatenate((_l, [_label[i]])) if _l != list([]) else [_label[i]]
    return _d, _l


def windowed_with_rr(_sig, _label, _index):
    _d, _l = [], []
    for i in range(1, len(_label) - 1):        # last one no count
        _from, _to = _index[i] - 128, _index[i] + 128
        if _from < 0:
            continue
        else:
            pre_rr = _index[i] - _index[i - 1]
            post_rr = _index[i + 1] - _index[i]
            d_rr = post_rr - pre_rr
            buffer = np.concatenate((_sig[_from: _to], [pre_rr], [post_rr], [d_rr]), axis=-1)

        if len(buffer) == 259:
            _d = np.concatenate(((_d, [buffer]))) if _d != list([]) else [buffer]
            _l = np.concatenate((_l, [_label[i]])) if _l != list([]) else [_label[i]]
    return _d, _l


def windowed_no_rr(_sig, _label, _index):
    _d, _l = [], []
    for i in range(len(_label) - 1):        # last one no count
        _from, _to = _index[i] - 128, _index[i] + 128
        if _from < 0:
            continue
        else:
            buffer = _sig[_from: _to]

        if len(buffer) == 256:
            _d = np.concatenate(((_d, [buffer]))) if _d != list([]) else [buffer]
            _l = np.concatenate((_l, [_label[i]])) if _l != list([]) else [_label[i]]
    return _d, _l


def windowed_v0(_sig, _label, _index):
    _d, _l = [], []
    for i in range(len(_label) - 1):        # last one no count
        _from, _to = _index[i] - 128, _index[i] + 128
        if _from < 0:
            continue
        else:
            buffer = _sig[_from: _to]

        if len(buffer) == 256:
            _d = np.concatenate(((_d, [buffer]))) if _d != list([]) else [buffer]
            _l = np.concatenate((_l, [_label[i]])) if _l != list([]) else [_label[i]]
    return _d, _l


def from_dict_to_array(dict):
    array, data_size = [], []
    for i in np.array(list(dict.keys())):
        buffer = np.array(dict[str(i)])
        array = np.concatenate((array, buffer)) if array != list([]) else buffer
    return array


def kiranyez_cmds(data_set, label):
    n_class = len(np.where(label == 0)[0])
    s_class = len(np.where(label == 1)[0])
    v_class = len(np.where(label == 2)[0])
    f_class = np.where(label == 3)
    q_class = np.where(label == 4)

    n_class = np.random.choice(a=int(n_class), size=75, replace=False)
    s_class = np.random.choice(a=int(s_class), size=75, replace=False)
    v_class = np.random.choice(a=int(v_class), size=75, replace=False)

    cmds = np.concatenate((data_set[n_class], data_set[s_class], data_set[v_class], data_set[f_class], data_set[q_class]))
    cmls = np.concatenate((label[n_class], label[s_class], label[v_class], label[f_class], label[q_class]))
    return cmds, cmls


def get_data(cmds_list, psds_list):

    # test_list = ['100']
    # _cmds, _cmls = read_data(list=test_list)
    _cmds, _cmls = read_data(list=cmds_list)
    _cmds = from_dict_to_array(dict=_cmds)
    _cmls = from_dict_to_array(dict=_cmls)
    _cmds, _cmls = kiranyez_cmds(data_set=_cmds, label=_cmls)  # 245 data

    _psds, _psls = read_data(list=psds_list, sampfrom=0, sampto=rd._5_mins)
    _psds = from_dict_to_array(dict=_psds)
    _psls = from_dict_to_array(dict=_psls)

    train_X, train_Y = np.concatenate((_cmds, _psds)), np.concatenate((_cmls, _psls))

    test_X, test_Y = read_data(list=psds_list, sampfrom=rd._5_mins, sampto=None)
    test_X = from_dict_to_array(dict=test_X)
    test_Y = from_dict_to_array(dict=test_Y)

    return train_X, train_Y, test_X, test_Y
############################################################


def save_img_data(_list):
    for i in _list:
        record_name = rd.local_record_dir + str(i)
        print('record_name: ', i)
        sig, label, beats_index = read(record_name=record_name)

        windowed_save_img(_sig=sig, _label=label, _index=beats_index, name=i)

data_len = 128
dpi = 100
def windowed_save_img(_sig, _label, _index, name):
    for i in range(1, len(_label) - 1):        # last one no count
        pre_rr = int(0.433 * (_index[i] - _index[i - 1]))
        post_rr = int(0.590 * (_index[i + 1] - _index[i]))
        _from, _to = _index[i] - pre_rr, _index[i] + post_rr
        if _from < 0:
            continue
        else:
            buffer = _sig[_from: _to]
            pictured(_input=buffer, name=name, count=i)


def pictured(_input, count, name):
    # store sample figures
    _64x64 = (0.7, 0.7)
    _128x128 = (1.4, 1.4)
    store_name = str(name) + '_' + str(count)
    plt.figure(figsize=_64x64)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.1, hspace=0.1)
    plt.plot(_input, color='grey', linewidth=0.5)
    plt.savefig(store_name, dpi=dpi)
    plt.close()
    print(store_name + ' img saved')


def pic_to_array(filename):
    img = Image.open(fp=filename + '.png')
    data = np.array(list(Image.Image.getdata(img)))[:, 0]
    data = np.reshape(data, (70, 70))[3:-3, 3:-3]
    if data.shape != (64, 64):
        logging.warning(msg='data shape warning {0}'.format([data.shape]))
    return data
    pass


def output_csv(data_set, label_set, num):
    store_name = 'data_set_record_' + str(num) + '_img'
    utilise.write_nd_data_v1(_input=data_set, filename=store_name)
    store_name = 'label_set_record_' + str(num) + '_img'
    utilise.write_nd_data_v0(_input=label_set, filename=store_name)


def read_pic_data(_list, sampfrom=0, sampto=None):

    for i in _list:
        data_set, label_set = [], []
        record_name = rd.local_record_dir + str(i)
        print('record_name: ', i)
        _, label, _ = read(record_name=record_name, sampfrom=sampfrom, sampto=sampto); _, shift, _ = read(record_name=record_name); shift = len(shift) - len(label) + 1
        del _

        print('this data contains ', len(label), ' heartbeat imgs')
        for j in range(shift, len(label) - 1):
            filename = str(i) + '_' + str(j)
            print('running img ', filename)
            array = pic_to_array(filename=filename).flatten()
            data_set = np.concatenate((data_set, [array])) if data_set != list([]) else [array]
            label_set = np.concatenate((label_set, [label[j]])) if label_set != list([]) else [label[j]]

        print(np.shape(data_set), np.shape(label_set))
        output_csv(data_set=data_set, label_set=label_set, num=i)

    return data_set, label_set
############################################################

def make_kiranyez_data(cmds_list, psds_list):
    _cmds, _cmls = [], []
    # load data
    for i in cmds_list:
        store_name = 'data_set_record_' + str(i) + '_img'
        data_set = np.loadtxt(fname=store_name+'.csv', delimiter=',')
        store_name = 'label_set_record_' + str(i) + '_img'
        label_set = np.loadtxt(fname=store_name+'.csv', delimiter=',')
        _cmds = np.concatenate((_cmds, data_set)) if _cmds != list([]) else data_set
        _cmls = np.concatenate((_cmls, label_set)) if _cmds != list([]) else label_set

    _cmds, _cmls = kiranyez_cmds(data_set=_cmds, label=_cmls)  # 245 data

    _psds, _psls = read_pic_data(_list=psds_list, sampfrom=0, sampto=_5_mins)

    train_X, train_Y = np.concatenate((_cmds, _psds)), np.concatenate((_cmls, _psls))

    test_X, test_Y = read_pic_data(_list=rd.all_44_record_list, sampfrom=_5_mins)

    return train_X, train_Y, test_X, test_Y


def mask_top_k(weights, bias, sparsity=0.5, filename=None):
    k = int(len(bias) * (1 - sparsity))
    thres_w = np.reshape(np.sort(np.abs(weights))[:, -k], [-1, 1])
    thres_b = np.sort(np.abs(bias))[-k]
    mask_w_p = weights > thres_w; mask_w_n = weights < -thres_w; mask_w = mask_w_p + mask_w_n
    mask_b_p = bias > thres_b; mask_b_n = bias < - thres_b; mask_b = mask_b_p + mask_b_n
    if filename != None:
        utilise.write_nd_data_v1(_input=weights * mask_w, filename='./results/'+str(filename))
    return mask_w, mask_b


def prune_top_k(weights, bias, sparsity=0.5):
    k = int(len(bias) * (1 - sparsity))
    thres_w = np.reshape(np.sort(np.abs(weights)[:, -k]), [-1, 1])
    thres_b = np.sort(np.abs(bias))[-k]
    weights = weights[weights > thres_w]
    bias = bias[bias > thres_b]
    return weights, bias

w_list = ['w_enc_h1', 'w_enc_h2', 'w_enc_h3', 'w_dec_h1', 'w_dec_h2', 'w_dec_h3']
mask_w_list = ['mask_w_enc_h1', 'mask_w_enc_h2', 'mask_w_enc_h3', 'mask_w_dec_h1', 'mask_w_dec_h2', 'mask_w_dec_h3']
b_list = ['b_enc_h1', 'b_enc_h2', 'b_enc_h3', 'b_dec_h1', 'b_dec_h2', 'b_dec_h3']
mask_b_list = ['mask_b_enc_h1', 'mask_b_enc_h2', 'mask_b_enc_h3', 'mask_b_dec_h1', 'mask_b_dec_h2', 'mask_b_dec_h3']
def restore_v1(para_dict, mask_dict, w_list=w_list, mask_w_list=mask_w_list, b_list=b_list, mask_b_list=mask_b_list):
    w_list = np.reshape(w_list, newshape=[-1, 1])
    mask_w_list = np.reshape(mask_w_list, newshape=[-1, 1])
    loop_w_list = np.concatenate((w_list, mask_w_list), axis=-1)
    for i, j in loop_w_list:
        N, M = np.shape(para_dict[i])
        w_r = np.ones(shape=[N, M], dtype=np.float32)
        for k in range(N):
            t = 0
            for l in range(M):
                if mask_dict[j][k, l] != True:
                    w_r[k, l] *= 0
                elif mask_dict[j][k, l] == True:
                    w_r[k, l] *= para_dict[i][k, t]
                    t += 1
                    pass
                pass
            pass
        para_dict[i] = w_r

    b_list = np.reshape(b_list, newshape=[-1, 1])
    mask_b_list = np.reshape(mask_b_list, newshape=[-1, 1])
    loop_b_list = np.concatenate((b_list, mask_b_list), axis=-1)
    for i, j in loop_b_list:
        M = len(para_dict[i])
        b_r = np.ones(shape=[M], dtype=np.float32)
        t = 0
        for k in range(M):
            if mask_dict[j][k] != True:
                b_r[k] *= 0
            elif mask_dict[j][k] == True:
                b_r[k] *= para_dict[i][t]
                t += 1
                pass
            pass
        para_dict[i] = b_r
        pass
    return para_dict




def restore_v0(para_dict, mask_dict):
    w_enc = para_dict['w_enc']
    mask_w_enc = mask_dict['mask_w_enc']
    N, M = np.shape(w_enc)
    w_r_enc = np.ones(shape=[N, M], dtype=np.float32)
    for i in range(N):
        t = 0
        for j in range(M):
            if mask_w_enc[i, j] != True:
                w_r_enc[i, j] *= 0
            elif mask_w_enc[i, j] == True:
                w_r_enc[i, j] *= w_enc[i, t]
                t += 1

    w_dec = para_dict['w_dec']
    mask_w_dec = mask_dict['mask_w_dec']
    N, M = np.shape(w_dec)
    w_r_dec = np.ones(shape=[N, M], dtype=np.float32)
    for i in range(N):
        t = 0
        for j in range(M):
            if mask_w_dec[i, j] != True:
                w_r_dec[i, j] *= 0
            elif mask_w_dec[i, j] == True:
                w_r_dec[i, j] *= w_dec[i, t]
                t += 1

    b_enc = para_dict['b_enc']
    mask_b_enc = mask_dict['mask_b_enc']
    M = len(b_enc)
    b_r_enc = np.ones(shape=[M], dtype=np.float32)
    t = 0
    for j in range(M):
        if mask_b_enc[j] != True:
            b_r_enc[j] *= 0
        elif mask_b_enc[j] == True:
            b_r_enc[j] *= b_enc[t]
            t += 1

    b_dec = para_dict['b_dec']
    mask_b_dec = mask_dict['mask_b_dec']
    M = len(b_dec)
    b_r_dec = np.ones(shape=[M], dtype=np.float32)
    t = 0
    for j in range(M):
        if mask_b_dec[j] != True:
            b_r_dec[j] *= 0
        elif mask_b_dec[j] == True:
            b_r_dec[j] *= b_dec[t]
            t += 1

    reco_dict = {'w_enc': w_r_enc, 'w_dec': w_r_dec, 'b_enc': b_r_enc, 'b_dec': b_r_dec}
    return reco_dict
    pass
