import numpy as np
from numpy import linalg as LA
from settingup import args
import sys
import copy

if args.seed:
    seed = args.seed
    np.random.seed(seed=seed)

def computeBfVector(theta, f_c=args.F_wave, n_bs=args.N_bs):
    """
    :param theta:
    :param f_c:
    :param n_bs:
    :return: BF vector, the norm square is 1.
    """
    c = 3e8
    wavelength = c / f_c

    d = wavelength / 2.
    k = 2. * np.pi / wavelength

    exponent = 1j * k * d * np.cos(theta) * np.arange(n_bs)

    bf = 1. / np.sqrt(n_bs) * np.exp(exponent)

    # Test the norm square... is it equal to unity? YES.
    #   norm_f_sq = LA.norm(f, ord=2) ** 2
    #   print(norm_f_sq)
    # print("bf vector shape: ", bf.shape)
    return bf

def computeChannel(multipaths=args.multipaths, n_bs=args.N_bs):
    """
    :param multipaths:
    :param n_bs:
    :return: a single channel model
    """
    # alpha denotes the channel gain of l paths, obeyed N(0,1)
    alpha = np.random.normal(size=multipaths)
    # the normalized receive array response vectors with AoA
    theta = (np.random.random() - 0.5) * np.pi

    bfvector = computeBfVector(theta=theta, n_bs=n_bs)

    channel = np.sqrt(n_bs / multipaths) * alpha.reshape(-1, 1) * bfvector

    channel = np.sum(channel, axis=0)
    # print("channel shape: ", channel.shape)

    return channel

def computeDigitalBf(H):
    assert isinstance(H, np.ndarray)
    # using Zero-forcing method
    res = H.conj().T
    res = np.matmul(res, H)
    res = np.linalg.inv(res)
    res = np.matmul(H, res)
    print("DigitalBf shape: ", res.shape)
    return res

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



def userGrouping(groups, usernums):
    """
    :param groups: a scalar
    :param usernums: a scalar
    :return: a dictionary, the key is the index of groups which value represents the member of the group
    """
    users = np.asarray([computeChannel() for i in range(usernums)])
    groups_dict = {i: [users[i]] for i in range(usernums)}
    while len(groups_dict) > groups:
        max_val = [None, sys.float_info.min]
        N = len(groups_dict)
        for i in range(N - 2, -1, -1):
            temp = computeGroupCHCorre(groups_dict[N-1], groups_dict[i])
            if temp >= max_val[1]:
                max_val[0] = i
                max_val[1] = temp
        if max_val[0] is not None:
            curr = groups_dict.pop(N-1)
            groups_dict[max_val[0]].extend(curr)
        else:
            print("Grouping Error")
            assert None
    return groups_dict

def transformState(groups_dict, N_bs=args.N_bs):
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
    channel_norm = copy.deepcopy(_channel_norm)
    if LA.norm(channel_norm[0]) == 0:
        return None
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


if __name__ == '__main__':
    g = userGrouping(7, 20)
    channel = transformState(g)
    latter = latterPartState(channel)
    # while latter is not None:
    #     latter = latterPartState(latter)
