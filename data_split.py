import numpy as np


#TODO: to do six labels classification
def data_split(data_set, label_set, train_rate):
    # good = ['N', 'A', 'V', 'L', 'P', 'R']
    N_Label_ids = np.where(label_set == [0])
    A_Label_ids = np.where(label_set == [1])
    V_Label_ids = np.where(label_set == [2])
    L_Label_ids = np.where(label_set == [3])
    P_Label_ids = np.where(label_set == [4])
    R_Label_ids = np.where(label_set == [5])

    N_dataset, A_dataset, V_dataset, L_dataset, P_dataset, R_dataset = data_set[N_Label_ids], data_set[A_Label_ids], data_set[V_Label_ids], data_set[L_Label_ids], data_set[P_Label_ids], data_set[R_Label_ids]
    N_labelset, A_labelset, V_labelset, L_labelset, P_labelset, R_labelset = label_set[N_Label_ids], label_set[A_Label_ids], label_set[V_Label_ids], label_set[L_Label_ids], label_set[P_Label_ids], label_set[R_Label_ids]

    N_train_dataset, A_train_dataset, V_train_dataset, L_train_dataset, P_train_dataset, R_train_dataset = \
        N_dataset[0: int(len(N_dataset) * train_rate)], \
        A_dataset[0: int(len(A_dataset) * train_rate)], \
        V_dataset[0: int(len(V_dataset) * train_rate)], \
        L_dataset[0: int(len(L_dataset) * train_rate)], \
        P_dataset[0: int(len(P_dataset) * train_rate)], \
        R_dataset[0: int(len(R_dataset) * train_rate)]
    train_X = np.vstack((N_train_dataset, A_train_dataset, V_train_dataset, L_train_dataset, P_train_dataset, R_train_dataset))

    N_train_labelset, A_train_labelset, V_train_labelset, L_train_labelset, P_train_labelset, R_train_labelset = \
        N_labelset[0: int(len(N_labelset) * train_rate)], \
        A_labelset[0: int(len(A_labelset) * train_rate)], \
        V_labelset[0: int(len(V_labelset) * train_rate)], \
        L_labelset[0: int(len(L_labelset) * train_rate)], \
        P_labelset[0: int(len(P_labelset) * train_rate)], \
        R_labelset[0: int(len(R_labelset) * train_rate)]
    train_Y = np.concatenate((N_train_labelset, A_train_labelset, V_train_labelset, L_train_labelset, P_train_labelset, R_train_labelset))

    N_test_dataset, A_test_dataset, V_test_dataset, L_test_dataset, P_test_dataset, R_test_dataset = \
        N_dataset[int(len(N_dataset) * train_rate):], \
        A_dataset[int(len(A_dataset) * train_rate):], \
        V_dataset[int(len(V_dataset) * train_rate):], \
        L_dataset[int(len(L_dataset) * train_rate):], \
        P_dataset[int(len(P_dataset) * train_rate):], \
        R_dataset[int(len(R_dataset) * train_rate):]
    test_X = np.vstack((N_test_dataset, A_test_dataset, V_test_dataset, L_test_dataset, P_test_dataset, R_test_dataset))

    N_test_labelset, A_test_labelset, V_test_labelset, L_test_labelset, P_test_labelset, R_test_labelset = \
        N_labelset[int(len(N_labelset) * train_rate):], \
        A_labelset[int(len(A_labelset) * train_rate):], \
        V_labelset[int(len(V_labelset) * train_rate):], \
        L_labelset[int(len(L_labelset) * train_rate):], \
        P_labelset[int(len(P_labelset) * train_rate):], \
        R_labelset[int(len(R_labelset) * train_rate):]
    test_Y = np.concatenate((N_test_labelset, A_test_labelset, V_test_labelset, L_test_labelset, P_test_labelset, R_test_labelset))

    return train_X, train_Y, test_X, test_Y
def cross_validation(data_set, label_set, fold=10):
    N_Label_ids = np.where(label_set == [0])
    A_Label_ids = np.where(label_set == [1])
    V_Label_ids = np.where(label_set == [2])
    L_Label_ids = np.where(label_set == [3])
    P_Label_ids = np.where(label_set == [4])
    R_Label_ids = np.where(label_set == [5])

    N_dataset, A_dataset, V_dataset, L_dataset, P_dataset, R_dataset = data_set[N_Label_ids], data_set[A_Label_ids], data_set[V_Label_ids], data_set[L_Label_ids], data_set[P_Label_ids], data_set[R_Label_ids]
    N_labelset, A_labelset, V_labelset, L_labelset, P_labelset, R_labelset = label_set[N_Label_ids], label_set[A_Label_ids], label_set[V_Label_ids], label_set[L_Label_ids], label_set[P_Label_ids], label_set[R_Label_ids]

    num_N = np.shape(N_Label_ids)[1]; seg_N = int(num_N // fold)
    num_A = np.shape(A_Label_ids)[1]; seg_A = int(num_A // fold)
    num_V = np.shape(V_Label_ids)[1]; seg_V = int(num_V // fold)
    num_L = np.shape(L_Label_ids)[1]; seg_L = int(num_L // fold)
    num_P = np.shape(P_Label_ids)[1]; seg_P = int(num_P // fold)
    num_R = np.shape(R_Label_ids)[1]; seg_R = int(num_R // fold)

    new_data_set = np.zeros(fold)
    new_label_set = np.zeros(fold)
    for i in range(fold):
        data_set = np.concatenate((np.array(N_dataset[i * seg_N: (i + 1) * seg_N]),
                                   np.array(A_dataset[i * seg_A: (i + 1) * seg_A]),
                                   np.array(V_dataset[i * seg_V: (i + 1) * seg_V]),
                                   np.array(L_dataset[i * seg_L: (i + 1) * seg_L]),
                                   np.array(P_dataset[i * seg_P: (i + 1) * seg_P]),
                                   np.array(R_dataset[i * seg_R: (i + 1) * seg_R])))
        new_data_set = np.concatenate((new_data_set, data_set)) if i != 0 else data_set

        label_set = np.concatenate((N_labelset[i * seg_N: (i+1) * seg_N],
                                    A_labelset[i * seg_A: (i+1) * seg_A],
                                    V_labelset[i * seg_V: (i+1) * seg_V],
                                    L_labelset[i * seg_L: (i+1) * seg_L],
                                    P_labelset[i * seg_P: (i+1) * seg_P],
                                    R_labelset[i * seg_R: (i+1) * seg_R]))
        new_label_set = np.concatenate((new_label_set, label_set)) if i != 0 else label_set

    return new_data_set, new_label_set.reshape([fold, -1])
    pass


def _data_split_6(data_set, label_set, train_rate):
    """
    good6 = ['A', 'L', 'N', 'P', 'R', 'V']
    """
    A_Label_ids, L_Label_ids, N_Label_ids, P_Label_ids, R_Label_ids, V_Label_ids = np.where(label_set == 0), \
                                                                                   np.where(label_set == 1), \
                                                                                   np.where(label_set == 2), \
                                                                                   np.where(label_set == 3), \
                                                                                   np.where(label_set == 4), \
                                                                                   np.where(label_set == 5)

    A_dataset, L_dataset, N_dataset, P_dataset, R_dataset, V_dataset, = data_set[A_Label_ids], \
                                                                        data_set[L_Label_ids], \
                                                                        data_set[N_Label_ids], \
                                                                        data_set[P_Label_ids], \
                                                                        data_set[R_Label_ids], \
                                                                        data_set[V_Label_ids]
    A_labelset, L_labelset, N_labelset, P_labelset, R_labelset, V_labelset, = label_set[A_Label_ids], \
                                                                              label_set[L_Label_ids], \
                                                                              label_set[N_Label_ids], \
                                                                              label_set[P_Label_ids], \
                                                                              label_set[R_Label_ids], \
                                                                              label_set[V_Label_ids]

    A_train_dataset, L_train_dataset, N_train_dataset, P_train_dataset, R_train_dataset, V_train_dataset = A_dataset[0: int(len(A_dataset) * train_rate)], \
                                                                                                           L_dataset[0: int(len(L_dataset) * train_rate)], \
                                                                                                           N_dataset[0: int(len(N_dataset) * train_rate)], \
                                                                                                           P_dataset[0: int(len(P_dataset) * train_rate)], \
                                                                                                           R_dataset[0: int(len(R_dataset) * train_rate)], \
                                                                                                           V_dataset[0: int(len(V_dataset) * train_rate)],
    train_X = np.concatenate((A_train_dataset, L_train_dataset, N_train_dataset, P_train_dataset, R_train_dataset, V_train_dataset))

    A_train_labelset, L_train_labelset, N_train_labelset, P_train_labelset, R_train_labelset, V_train_labelset = A_labelset[0: int(len(A_labelset) * train_rate)], \
                                                                                                                 L_labelset[0: int(len(L_labelset) * train_rate)], \
                                                                                                                 N_labelset[0: int(len(N_labelset) * train_rate)], \
                                                                                                                 P_labelset[0: int(len(P_labelset) * train_rate)], \
                                                                                                                 R_labelset[0: int(len(R_labelset) * train_rate)], \
                                                                                                                 V_labelset[0: int(len(V_labelset) * train_rate)]
    train_Y = np.concatenate((A_train_labelset, L_train_labelset, N_train_labelset, P_train_labelset, R_train_labelset, V_train_labelset))

    A_test_dataset, L_test_dataset, N_test_dataset, P_test_dataset, R_test_dataset, V_test_dataset = A_dataset[int(len(A_dataset) * train_rate):], \
                                                                                                     L_dataset[int(len(L_dataset) * train_rate):], \
                                                                                                     N_dataset[int(len(N_dataset) * train_rate):], \
                                                                                                     P_dataset[int(len(P_dataset) * train_rate):], \
                                                                                                     R_dataset[int(len(R_dataset) * train_rate):], \
                                                                                                     V_dataset[int(len(V_dataset) * train_rate):]
    test_X = np.concatenate((A_test_dataset, L_test_dataset, N_test_dataset, P_test_dataset, R_test_dataset, V_test_dataset))

    A_test_labelset, L_test_labelset, N_test_labelset, P_test_labelset, R_test_labelset, V_test_labelset = A_labelset[int(len(A_labelset) * train_rate):], \
                                                                                                           L_labelset[int(len(L_labelset) * train_rate):], \
                                                                                                           N_labelset[int(len(N_labelset) * train_rate):], \
                                                                                                           P_labelset[int(len(P_labelset) * train_rate):], \
                                                                                                           R_labelset[int(len(R_labelset) * train_rate):], \
                                                                                                           V_labelset[int(len(V_labelset) * train_rate):],
    test_Y = np.concatenate((A_test_labelset, L_test_labelset, N_test_labelset, P_test_labelset, R_test_labelset, V_test_labelset))

    return train_X, train_Y, test_X, test_Y
def _data_split_7(data_set, label_set, train_rate):
    """
    good7 = ['N', 'V', '/', 'L', '!', 'R', 'E']
    """
    NOR_ids, PVC_ids, PACE_ids, LBB_ids, FLW_ids, RBB_ids, VESC_ids = np.where(label_set == 0), \
                                                                      np.where(label_set == 1), \
                                                                      np.where(label_set == 2), \
                                                                      np.where(label_set == 3), \
                                                                      np.where(label_set == 4), \
                                                                      np.where(label_set == 5), \
                                                                      np.where(label_set == 6)
    NOR_ds, PVC_ds, PACE_ds, LBB_ds, FLW_ds, RBB_ds, VESC_ds = data_set[NOR_ids ], \
                                                               data_set[PVC_ids ], \
                                                               data_set[PACE_ids], \
                                                               data_set[LBB_ids ], \
                                                               data_set[FLW_ids ], \
                                                               data_set[RBB_ids ], \
                                                               data_set[VESC_ids]

    NOR_ls, PVC_ls, PACE_ls, LBB_ls, FLW_ls, RBB_ls, VESC_ls = label_set[NOR_ids ], \
                                                               label_set[PVC_ids ], \
                                                               label_set[PACE_ids], \
                                                               label_set[LBB_ids ], \
                                                               label_set[FLW_ids ], \
                                                               label_set[RBB_ids ], \
                                                               label_set[VESC_ids]
    NOR_train_ds, PVC_train_ds, PACE_train_ds, LBB_train_ds, FLW_train_ds, RBB_train_ds, VESC_train_ds = NOR_ds[0: int(len(NOR_ds) * train_rate)], \
                                                                                                         PVC_ds[0: int(len(PVC_ds) * train_rate)], \
                                                                                                         PACE_ds[0: int(len(PACE_ds) * train_rate)], \
                                                                                                         LBB_ds[0: int(len(LBB_ds) * train_rate)], \
                                                                                                         FLW_ds[0: int(len(FLW_ds) * train_rate)], \
                                                                                                         RBB_ds[0: int(len(RBB_ds) * train_rate)], \
                                                                                                         VESC_ds[0: int(len(VESC_ds) * train_rate)]
    train_X = np.concatenate((NOR_train_ds, PVC_train_ds, PACE_train_ds, LBB_train_ds, FLW_train_ds, RBB_train_ds, VESC_train_ds))

    NOR_train_ls, PVC_train_ls, PACE_train_ls, LBB_train_ls, FLW_train_ls, RBB_train_ls, VESC_train_ls = NOR_ls[0: int(len(NOR_ls) * train_rate)], \
                                                                                                         PVC_ls[0: int(len(PVC_ls) * train_rate)], \
                                                                                                         PACE_ls[0: int(len(PACE_ls) * train_rate)], \
                                                                                                         LBB_ls[0: int(len(LBB_ls) * train_rate)], \
                                                                                                         FLW_ls[0: int(len(FLW_ls) * train_rate)], \
                                                                                                         RBB_ls[0: int(len(RBB_ls) * train_rate)], \
                                                                                                         VESC_ls[0: int(len(VESC_ls) * train_rate)]
    train_Y = np.concatenate((NOR_train_ls, PVC_train_ls, PACE_train_ls, LBB_train_ls, FLW_train_ls, RBB_train_ls, VESC_train_ls))

    NOR_test_ds, PVC_test_ds, PACE_test_ds, LBB_test_ds, FLW_test_ds, RBB_test_ds, VESC_test_ds = NOR_ds[int(len(NOR_ds) * train_rate):], \
                                                                                                  PVC_ds[int(len(PVC_ds) * train_rate):], \
                                                                                                  PACE_ds[int(len(PACE_ds) * train_rate):], \
                                                                                                  LBB_ds[int(len(LBB_ds) * train_rate):], \
                                                                                                  FLW_ds[int(len(FLW_ds) * train_rate):], \
                                                                                                  RBB_ds[int(len(RBB_ds) * train_rate):], \
                                                                                                  VESC_ds[int(len(VESC_ds) * train_rate):]
    test_X = np.concatenate((NOR_test_ds, PVC_test_ds, PACE_test_ds, LBB_test_ds, FLW_test_ds, RBB_test_ds, VESC_test_ds))

    NOR_test_ls, PVC_test_ls, PACE_test_ls, LBB_test_ls, FLW_test_ls, RBB_test_ls, VESC_test_ls = NOR_ls[int(len(NOR_ls) * train_rate):], \
                                                                                                  PVC_ls[int(len(PVC_ls) * train_rate):], \
                                                                                                  PACE_ls[int(len(PACE_ls) * train_rate):], \
                                                                                                  LBB_ls[int(len(LBB_ls) * train_rate):], \
                                                                                                  FLW_ls[int(len(FLW_ls) * train_rate):], \
                                                                                                  RBB_ls[int(len(RBB_ls) * train_rate):], \
                                                                                                  VESC_ls[int(len(VESC_ls) * train_rate):]

    test_Y = np.concatenate((NOR_test_ls, PVC_test_ls, PACE_test_ls, LBB_test_ls, FLW_test_ls, RBB_test_ls, VESC_test_ls))

    return train_X, train_Y, test_X, test_Y
def _data_split_5(data_set, label_set, train_rate):
    """
    good7 = ['N', 'V', '/', 'L', 'R']
    """
    NOR_ids, PVC_ids, PACE_ids, LBB_ids, RBB_ids = np.where(label_set == 0), \
                                                   np.where(label_set == 1), \
                                                   np.where(label_set == 2), \
                                                   np.where(label_set == 3), \
                                                   np.where(label_set == 4)
    NOR_ds, PVC_ds, PACE_ds, LBB_ds, RBB_ds = data_set[NOR_ids], \
                                              data_set[PVC_ids], \
                                              data_set[PACE_ids], \
                                              data_set[LBB_ids], \
                                              data_set[RBB_ids]

    NOR_ls, PVC_ls, PACE_ls, LBB_ls, RBB_ls = label_set[NOR_ids], \
                                              label_set[PVC_ids], \
                                              label_set[PACE_ids], \
                                              label_set[LBB_ids], \
                                              label_set[RBB_ids]
    NOR_train_ds, PVC_train_ds, PACE_train_ds, LBB_train_ds, RBB_train_ds = NOR_ds[0: int(len(NOR_ds) * train_rate)], \
                                                                            PVC_ds[0: int(len(PVC_ds) * train_rate)], \
                                                                            PACE_ds[0: int(len(PACE_ds) * train_rate)], \
                                                                            LBB_ds[0: int(len(LBB_ds) * train_rate)], \
                                                                            RBB_ds[0: int(len(RBB_ds) * train_rate)]
    train_X = np.concatenate((NOR_train_ds, PVC_train_ds, PACE_train_ds, LBB_train_ds, RBB_train_ds))

    NOR_train_ls, PVC_train_ls, PACE_train_ls, LBB_train_ls, RBB_train_ls = NOR_ls[0: int(len(NOR_ls) * train_rate)], \
                                                                            PVC_ls[0: int(len(PVC_ls) * train_rate)], \
                                                                            PACE_ls[0: int(len(PACE_ls) * train_rate)], \
                                                                            LBB_ls[0: int(len(LBB_ls) * train_rate)], \
                                                                            RBB_ls[0: int(len(RBB_ls) * train_rate)]
    train_Y = np.concatenate((NOR_train_ls, PVC_train_ls, PACE_train_ls, LBB_train_ls, RBB_train_ls))

    NOR_test_ds, PVC_test_ds, PACE_test_ds, LBB_test_ds, RBB_test_ds = NOR_ds[int(len(NOR_ds) * train_rate):], \
                                                                       PVC_ds[int(len(PVC_ds) * train_rate):], \
                                                                       PACE_ds[int(len(PACE_ds) * train_rate):], \
                                                                       LBB_ds[int(len(LBB_ds) * train_rate):], \
                                                                       RBB_ds[int(len(RBB_ds) * train_rate):]
    test_X = np.concatenate((NOR_test_ds, PVC_test_ds, PACE_test_ds, LBB_test_ds, RBB_test_ds))

    NOR_test_ls, PVC_test_ls, PACE_test_ls, LBB_test_ls, RBB_test_ls = NOR_ls[int(len(NOR_ls) * train_rate):], \
                                                                       PVC_ls[int(len(PVC_ls) * train_rate):], \
                                                                       PACE_ls[int(len(PACE_ls) * train_rate):], \
                                                                       LBB_ls[int(len(LBB_ls) * train_rate):], \
                                                                       RBB_ls[int(len(RBB_ls) * train_rate):]

    test_Y = np.concatenate((NOR_test_ls, PVC_test_ls, PACE_test_ls, LBB_test_ls, RBB_test_ls))

    return train_X, train_Y, test_X, test_Y


def _cross_validation(data_set, label_set, fold=10):
    """
    good6 = ['A', 'L', 'N', 'P', 'R', 'V']
    train_num = [2292, 7265, 67520, 6323, 6530, 6417]
    test_num = [254, 807, 7502, 702, 725, 712]
    """
    train_num = [2292, 7265, 67520, 6323, 6530, 6417]
    test_num = [254, 807, 7502, 702, 725, 712]
    data_len = np.shape(data_set)[-1]
    A_Label_ids, L_Label_ids, N_Label_ids, P_Label_ids, R_Label_ids, V_Label_ids = \
        np.array(np.where(label_set == 0)).squeeze(), \
        np.array(np.where(label_set == 1)).squeeze(), \
        np.array(np.where(label_set == 2)).squeeze(), \
        np.array(np.where(label_set == 3)).squeeze(), \
        np.array(np.where(label_set == 4)).squeeze(), \
        np.array(np.where(label_set == 5)).squeeze()
    # A:2544 L:8070 N:75003 P:7022 R:7254 V:7129
    # print(A_Label_ids.shape, L_Label_ids.shape, N_Label_ids.shape, np.shape(P_Label_ids), np.shape(R_Label_ids), np.shape(V_Label_ids))
    A_dataset, L_dataset, N_dataset, P_dataset, R_dataset, V_dataset = np.array(data_set[A_Label_ids]), \
                                                                       np.array(data_set[L_Label_ids]), \
                                                                       np.array(data_set[N_Label_ids]), \
                                                                       np.array(data_set[P_Label_ids]), \
                                                                       np.array(data_set[R_Label_ids]), \
                                                                       np.array(data_set[V_Label_ids])
    # print(A_dataset.shape, L_dataset.shape, N_dataset.shape, P_dataset.shape, R_dataset.shape, V_dataset.shape)
    A_labelset, L_labelset, N_labelset, P_labelset, R_labelset, V_labelset = np.array(label_set[A_Label_ids]), \
                                                                             np.array(label_set[L_Label_ids]), \
                                                                             np.array(label_set[N_Label_ids]), \
                                                                             np.array(label_set[P_Label_ids]), \
                                                                             np.array(label_set[R_Label_ids]), \
                                                                             np.array(label_set[V_Label_ids])

    train_data_set = np.zeros(fold)
    train_label_set = np.zeros(fold)
    test_data_set = np.zeros(fold)
    test_label_set = np.zeros(fold)
    for i in range(fold * 2):
        if i < 10:
            # random selection
            A_ds, L_ds, N_ds, P_ds, R_ds, V_ds = np.random.choice(2544,  train_num[0], replace=False), \
                                                 np.random.choice(8070,  train_num[1], replace=False), \
                                                 np.random.choice(75003, train_num[2], replace=False), \
                                                 np.random.choice(7022,  train_num[3], replace=False), \
                                                 np.random.choice(7254,  train_num[4], replace=False), \
                                                 np.random.choice(7129,  train_num[5], replace=False)
        else:
            A_ds, L_ds, N_ds, P_ds, R_ds, V_ds = np.random.choice(2544,  test_num[0], replace=False), \
                                                 np.random.choice(8070,  test_num[1], replace=False), \
                                                 np.random.choice(75003, test_num[2], replace=False), \
                                                 np.random.choice(7022,  test_num[3], replace=False), \
                                                 np.random.choice(7254,  test_num[4], replace=False), \
                                                 np.random.choice(7129,  test_num[5], replace=False)

        # assign randomly selected indexes
        A_dataset_b, L_dataset_b, N_dataset_b, P_dataset_b, R_dataset_b, V_dataset_b = \
            np.array(A_dataset[A_ds]), \
            np.array(L_dataset[L_ds]), \
            np.array(N_dataset[N_ds]), \
            np.array(P_dataset[P_ds]), \
            np.array(R_dataset[R_ds]), \
            np.array(V_dataset[V_ds])
        A_labelset_b, L_labelset_b, N_labelset_b, P_labelset_b, R_labelset_b, V_labelset_b = \
            np.array(A_labelset[A_ds]), \
            np.array(L_labelset[L_ds]), \
            np.array(N_labelset[N_ds]), \
            np.array(P_labelset[P_ds]), \
            np.array(R_labelset[R_ds]), \
            np.array(V_labelset[V_ds])

        data_set = np.concatenate((A_dataset_b, L_dataset_b, N_dataset_b, P_dataset_b, R_dataset_b, V_dataset_b))
        label_set = np.concatenate((A_labelset_b, L_labelset_b, N_labelset_b, P_labelset_b, R_labelset_b, V_labelset_b))
        if i < 10:
            train_data_set = np.concatenate((train_data_set, [data_set])) if i != 0 else [data_set]
            train_label_set = np.concatenate((train_label_set, [label_set])) if i != 0 else [label_set]
        else:
            test_data_set = np.concatenate((test_data_set, [data_set])) if i != 10 else [data_set]
            test_label_set = np.concatenate((test_label_set, [label_set])) if i != 10 else [label_set]

    # print(train_data_set.shape, train_label_set.shape, test_data_set.shape, test_label_set.shape)
    return train_data_set.reshape([fold, -1, data_len]), \
           train_label_set.reshape([fold, -1]), \
           test_data_set.reshape([fold, -1, data_len]), \
           test_label_set.reshape([fold, -1])
    pass

def _cross_validation_test(data_set, label_set, fold=10):
    """
    good6 = ['A', 'L', 'N', 'P', 'R', 'V']
    train_num = [2292, 7265, 67520, 6323, 6530, 6417]
    test_num = [254, 807, 7502, 702, 725, 712]
    """
    train_num = [90, 90, 90, 90, 90, 90]
    test_num = [10, 10, 10, 10, 10, 10]
    data_len = np.shape(data_set)[-1]
    A_Label_ids, L_Label_ids, N_Label_ids, P_Label_ids, R_Label_ids, V_Label_ids = np.array(np.where(label_set == [0])).squeeze(), \
                                                                                   np.array(np.where(label_set == [1])).squeeze(), \
                                                                                   np.array(np.where(label_set == [2])).squeeze(), \
                                                                                   np.array(np.where(label_set == [3])).squeeze(), \
                                                                                   np.array(np.where(label_set == [4])).squeeze(), \
                                                                                   np.array(np.where(label_set == [5])).squeeze()
    A_dataset, L_dataset, N_dataset, P_dataset, R_dataset, V_dataset = np.arange(0, 2560000).reshape([10000, 256]), \
                                                                       np.arange(0, 2560000).reshape([10000, 256]), \
                                                                       np.arange(0, 2560000).reshape([10000, 256]), \
                                                                       np.arange(0, 2560000).reshape([10000, 256]), \
                                                                       np.arange(0, 2560000).reshape([10000, 256]), \
                                                                       np.arange(0, 2560000).reshape([10000, 256])
    A_labelset, L_labelset, N_labelset, P_labelset, R_labelset, V_labelset = np.array(label_set[A_Label_ids]), \
                                                                             np.array(label_set[L_Label_ids]), \
                                                                             np.array(label_set[N_Label_ids]), \
                                                                             np.array(label_set[P_Label_ids]), \
                                                                             np.array(label_set[R_Label_ids]), \
                                                                             np.array(label_set[V_Label_ids])

    train_data_set = np.zeros(fold)
    train_label_set = np.zeros(fold)
    test_data_set = np.zeros(fold)
    test_label_set = np.zeros(fold)
    for i in range(fold * 2):
        if i < 10:
            # random selection
            A_ds, L_ds, N_ds, P_ds, R_ds, V_ds = np.random.choice(2544,  train_num[0], replace=False), \
                                                 np.random.choice(8070,  train_num[1], replace=False), \
                                                 np.random.choice(75003, train_num[2], replace=False), \
                                                 np.random.choice(7022,  train_num[3], replace=False), \
                                                 np.random.choice(7254,  train_num[4], replace=False), \
                                                 np.random.choice(7129,  train_num[5], replace=False)
        else:
            A_ds, L_ds, N_ds, P_ds, R_ds, V_ds = np.random.choice(2544,  test_num[0], replace=False), \
                                                 np.random.choice(8070,  test_num[1], replace=False), \
                                                 np.random.choice(75003, test_num[2], replace=False), \
                                                 np.random.choice(7022,  test_num[3], replace=False), \
                                                 np.random.choice(7254,  test_num[4], replace=False), \
                                                 np.random.choice(7129,  test_num[5], replace=False)

        # assign randomly selected indexes
        A_dataset, L_dataset, N_dataset, P_dataset, R_dataset, V_dataset = np.array(A_dataset)[A_ds], \
                                                                           np.array(L_dataset)[L_ds], \
                                                                           np.array(N_dataset)[N_ds], \
                                                                           np.array(P_dataset)[P_ds], \
                                                                           np.array(R_dataset)[R_ds], \
                                                                           np.array(V_dataset)[V_ds]
        A_labelset, L_labelset, N_labelset, P_labelset, R_labelset, V_labelset = np.array(A_labelset)[A_ds], \
                                                                                 np.array(L_labelset)[L_ds], \
                                                                                 np.array(N_labelset)[N_ds], \
                                                                                 np.array(P_labelset)[P_ds], \
                                                                                 np.array(R_labelset)[R_ds], \
                                                                                 np.array(V_labelset)[V_ds]

        data_set = np.concatenate((A_dataset, L_dataset, N_dataset, P_dataset, R_dataset, V_dataset))
        label_set = np.concatenate((A_labelset, L_labelset, N_labelset, P_labelset, R_labelset, V_labelset))
        if i < 10:
            train_data_set = np.concatenate((train_data_set, data_set)) if i != 0 else data_set
            train_data_set = np.concatenate((train_data_set, data_set)) if i != 0 else data_set
        else:
            test_data_set = np.concatenate((test_data_set, data_set)) if i != 0 else data_set
            test_label_set = np.concatenate((test_label_set, label_set)) if i != 0 else label_set

    return train_data_set.reshape([fold, -1, data_len]), \
           train_label_set.reshape([fold, -1]), \
           test_data_set.reshape([fold, -1, data_len]), \
           test_label_set.reshape([fold, -1])
    pass


def kf_shuffle(data_set, label_set, fold=10):
    """ kf: k-fold method """
    set_size = data_set.shape[1]
    for i in range(fold):
        random_idx = np.random.choice(a=set_size, size=set_size, replace=False)
        data_set[i] = data_set[i][random_idx]
        label_set[i] = label_set[i][random_idx]
    return data_set, label_set


def shuffle(data_set, label_set):
    """ data-set shuffle method """
    train_size = len(label_set)
    train_random_ind = np.random.choice(a=train_size, size=train_size, replace=False)
    new_data_set, new_label_set = data_set[train_random_ind], label_set[train_random_ind]
    del data_set, label_set
    return new_data_set, new_label_set

