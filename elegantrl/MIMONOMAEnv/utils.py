import numpy as np
from numpy import linalg as LA
from settingup import args_channel
import sys
import copy

if args_channel.seed:
    np.random.seed(seed=args_channel.seed)


def computeBfVector(theta, f_c=args_channel.F_wave, N_bs=args_channel.N_bs):
    """
    :param theta:
    :param f_c:
    :param N_bs:
    :return: BF vector, the norm square is 1.
    """

    assert 0 <= theta < 2 * np.pi or - np.pi / 2 <= theta <= np.pi / 2

    c = 3e8
    wavelength = c / f_c

    d = wavelength / 2.
    k = 2. * np.pi / wavelength

    exponent = 1j * k * d * np.cos(theta) * np.arange(N_bs)

    bf = 1. / np.sqrt(N_bs) * np.exp(exponent)

    # Test the norm square... is it equal to unity? YES.
    #   norm_f_sq = LA.norm(f, ord=2) ** 2
    #   print(norm_f_sq)
    # print("bf vector shape: ", bf.shape)
    return bf


def computeDigitalBf(H):
    assert isinstance(H, np.ndarray)
    # using Zero-forcing method
    res = H.conj().T
    res = np.matmul(res, H)
    res = np.linalg.inv(res)
    res = np.matmul(H, res)
    # print("DigitalBf shape: ", res.shape)
    return res


def computeChannel(multipaths=args_channel.multipaths, n_bs=args_channel.N_bs):
    """
    :param multipaths:
    :param n_bs:
    :return: a single channel model
    """
    # alpha denotes the channel gain of l paths, obeyed N(0,1)
    alpha = np.random.normal(size=multipaths)
    # the normalized receive array response vectors with AoA
    theta = (np.random.random() - 0.5) * np.pi

    bfvector = computeBfVector(theta=theta, N_bs=n_bs)

    channel = np.sqrt(n_bs / multipaths) * alpha.reshape(-1, 1) * bfvector

    channel = np.sum(channel, axis=0)
    # print("channel shape: ", channel.shape)

    return channel


def userGrouping(groups, user_numbers):
    """
    :param groups: a scalar
    :param user_numbers: a scalar
    :return: a dictionary, the key is the index of groups which value represents the member of the group
    """
    users = np.asarray([computeChannel() for i in range(user_numbers)])
    groups_dict = {i: [users[i]] for i in range(user_numbers)}
    while len(groups_dict) > groups:
        max_val = [None, sys.float_info.min]
        N = len(groups_dict)
        for i in range(N - 2, -1, -1):
            temp = computeGroupCHCorre(groups_dict[N - 1], groups_dict[i])
            if temp >= max_val[1]:
                max_val[0] = i
                max_val[1] = temp
        if max_val[0] is not None:
            curr = groups_dict.pop(N - 1)
            groups_dict[max_val[0]].extend(curr)
        else:
            print("Grouping Error")
            assert None
    return groups_dict


def transformState(groups_dict, N_bs=args_channel.N_bs):
    """
    :param groups_dict:
    :param N_bs:
    :return: channel_norm (channel, user_number, N_bs)
            channel: 0 is channel amplitude.
                    1 is channel angle.
                    2 is channel group flag.
    """
    user_number = 0
    for order in groups_dict:
        user_number += len(groups_dict[order])
    channel_amp = np.zeros((user_number, N_bs), dtype=np.float32)
    channel_angle = np.zeros((user_number, N_bs), dtype=np.float32)
    channel_groups = np.zeros((user_number, N_bs), dtype=np.float32)

    index = 0
    for order in groups_dict:
        for item in groups_dict[order]:
            channel_amp[index] = np.abs(item)
            channel_angle[index] = np.angle(item)
            channel_groups[index].fill(order)
            index += 1

    channel_norm = np.zeros((3, user_number, N_bs), dtype=np.float32)
    channel_norm[0] = channel_amp
    channel_norm[1] = channel_angle
    channel_norm[2] = channel_groups

    return channel_norm


def latterPartState(_channel_norm):
    """
    select next group to deal with, which means to set the value to zero.
    :param _channel_norm:
    :return:
    """
    channel_norm = copy.deepcopy(_channel_norm)
    if LA.norm(channel_norm[0]) == 0:
        return channel_norm
    index = 0
    while LA.norm(channel_norm[0][index]) == 0:
        index += 1
    latter_part_amp = np.zeros_like(channel_norm[0][index])
    latter_part_angle = np.zeros_like(channel_norm[1][index])

    group_order = channel_norm[2][index][0]

    while index < len(channel_norm[2]) and channel_norm[2][index][0] == group_order:
        channel_norm[0][index] = latter_part_amp
        channel_norm[1][index] = latter_part_angle
        index += 1
    return channel_norm


def computeCHCorrelation(h1, h2):
    """
    :param h1:
    :param h2:
    :return: a scalar, the return value is positive proportion to the similarity, 1 is the max
    """
    up = LA.norm(np.matmul(h1, h2.conj().T))
    bottom = LA.norm(h1) * LA.norm(h2)
    return up / bottom


def computeGroupCHCorre(group1, group2):
    """
    :param group1:
    :param group2:
    :return: a scalar, the return value is positive proportion to the similarity, 1 is the max
    """
    assert group1 and group2
    # compute Complete linkage distance
    differ = sys.float_info.max
    for item in group1:
        for item2 in group2:
            differ = min(differ, computeCHCorrelation(item, item2))
    return differ


def computeChABFGain(group, f_rf):
    """
    maybe the step reward of one group
    :param group:
    :param f_rf:
    :return: the sum of the norm of matmul(f_rf, channel)
    """
    gain = 0
    for item in group:
        gain = gain + LA.norm(np.matmul(f_rf.conjugate().T, item))
    return gain


def getFbb(F_rf, Pw, groups_dict):  # haven't been tested
    """
    :param F_rf: analog bf matrix (N_bs * N_rf)
    :param Pw: dict { group_order : power}
    :param groups_dict: dict { group_order : channel_list}
    :return: F_bb digital bf matrix (N_rf, N_rf)
    """
    groups_order_dict = {}
    for item in groups_dict:
        group = {}
        for elem in groups_dict[item]:
            elem = tuple(elem)
            group[elem] = LA.norm(np.matmul(F_rf.conjugate().T, np.asarray(elem)))
        group_ = sorted(group.items(), key=lambda d: d[1], reverse=True)
        groups_order_dict[item] = group_

    # formalize group_order_dict
    for i in groups_order_dict:
        for j in range(len(groups_order_dict[i])):
            groups_order_dict[i][j] = np.asarray(groups_order_dict[i][j][0])

    # compute digital bf vector with the largest bf gain in the associating group.
    H_bb = []
    for item in groups_order_dict:
        h = np.sqrt(Pw[item]) * np.matmul(F_rf.conjugate().T, groups_order_dict[item][0])
        H_bb.append(h)

    # compute digital bf matrix
    F_bb = computeDigitalBf(np.asarray(H_bb))

    # normalize f_bb

    for i in range(len(F_bb[0])):
        F_bb[:, i] = F_bb[:, i] / LA.norm(np.matmul(F_rf, F_bb[:, i]))

    return F_bb, groups_order_dict


def computeInterference(f_bb, F_rf, group, power, start_index=0, P_tol=args_channel.P_tol):  # haven't been tested
    """
    :param f_bb:
    :param F_rf:
    :param group: does this need sorted ? I think it does.
    :param power: the last one allocated power
    :param start_index:
    :param P_tol:
    :return:
    """
    bf_matrix = np.matmul(f_bb.conjugate().T, F_rf.conjugate().T)
    norm = lambda o: LA.norm(np.matmul(bf_matrix, group[o])) ** 2
    power_list = np.zeros(len(group))
    power_list[len(group)-1] = power
    if len(group) > 1:
        # compute power_list
        for index in range(len(group) - 2, start_index-1, -1):
            la = index+1
            P_sum = 0
            while la < len(group):
                P_sum = (norm(la) * power_list[la])
                la = la + 1
            power_list[index] = (P_tol + P_sum) / norm(index)

    interference = 0
    for i in range(start_index, len(group)):
        interference = interference + norm(i) * power_list[i]
    return interference, power_list


def computeSINR(group_order, user_order, F_rf, Pw, groups_dict, sigma=args_channel.sigma):
    F_bb, groups_order_dict = getFbb(F_rf=F_rf, Pw=Pw, groups_dict=groups_dict)
    _, power_list_intra = computeInterference(f_bb=F_bb[:, group_order], F_rf=F_rf,
                                              group=groups_order_dict[group_order],
                                              power=Pw[group_order], start_index=0)

    I_intra, _ = computeInterference(f_bb=F_bb[:, group_order], F_rf=F_rf,
                                     group=groups_order_dict[group_order],
                                     power=Pw[group_order], start_index=(user_order + 1))

    upp_bf_matrix = np.matmul(F_bb[:, group_order].conjugate().T, F_rf.conjugate().T)
    upp = LA.norm(np.matmul(upp_bf_matrix,
                  groups_order_dict[group_order][user_order])) ** 2 * power_list_intra[user_order]
    noise = LA.norm(upp_bf_matrix) ** 2 * sigma ** 2

    I_inter = 0
    for i in groups_order_dict:
        if i == group_order:
            continue
        I_inter = I_inter + computeInterference(f_bb=F_bb[:, i], F_rf=F_rf,
                                                group=groups_order_dict[i],
                                                power=Pw[i], start_index=0)[0]

    SINR_gu = upp / (I_intra + I_inter + noise)

    return SINR_gu


if __name__ == '__main__':
    f_rf1 = computeBfVector(1.5)
    g = userGrouping(7, 20)
    channel = transformState(g)
    latter = latterPartState(channel)
    l = np.vstack((channel, latter))
    gain1 = computeChABFGain(g[0], 2)
    print(gain1)
    t = np.array([[13, 2, 5], [4, 6, 8], [1, 2, 5], [5, 6, 8]])
    # while latter is not None:
    #     latter = latterPartState(latter)
