


"""
Convolutional Neural Networks for Patient- Specific ECG Classification by Serkan Kiranyaz et al.
"""

kir_good5 = ['N', 'e', 'L', 'R', 'j',  # 'N', 'S', 'V', 'F', 'Q'
         'S', 'A', 'a', 'J',
         'V', 'E',
         'F',
         '/', 'f', 'Q']

good4 = ['N', 'e', 'L', 'R', 'j',  # 'N', 'S', 'V', 'F'
         'A', 'a', 'J', 'S',
         'V', 'E',
         'F']

good7 = ['N', 'V', '/', 'L', '!', 'R', 'E']
good6 = ['A', 'L', 'N', '/', 'R', 'V']
good5 = ['N', 'V', '/', 'L', 'R']
# 'V'/PVC: prolonged premature ventricular contractions (PVCs) beats occasionally turn into
# a ventricular tachycardia (VT) or a ventricular fibrillation (VF) beats which can immediately
# lead to the heart failure.

#

from_100_to_124 = ['100', '101', '103', '105', '106',  # training data-set
                   '108', '109', '111', '112', '113',
                   '114', '115', '116', '117', '118',
                   '119', '121', '122', '123', '124']

from_200_to_234 = ['200', '201', '202', '203', '205', '207',
                   '208', '209', '210', '212', '213', '214',
                   '215', '219', '220', '221', '222', '223',
                   '228', '230', '231', '232', '233', '234']

all_48_record_list = from_100_to_124 + from_200_to_234 + ['102', '104', '107', '217']
all_44_record_list = from_100_to_124 + from_200_to_234

valid_VEB = [
    '200', '202', '210', '213', '214', '219', '221', '228', '231', '233', '234'
]  # for VEB detection

valid_SVEB = [
    '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]  # for SVEB detection


local_record_dir = '/Users/AppleUser/PycharmProjects/af_clsfy/base_evn/mitdb/'
server_record_dir = '/nfshome/chunyi/af_label_clasfy/database/mitdb/'

mitdb_fs = 360
_5_mins = mitdb_fs * 300


