import numpy as np
from numpy import linalg as LA

from utils import computeBfVector, computeChannel, \
     computeDigitalBf

import matplotlib.pyplot as plt

def beamSelect(group, beam_book):
    group_Frf = []
    for user in group:
        index = None
        gain = 0
        for i in range(len(beam_book)):
            if i in group_Frf:
                continue
            temp = LA.norm(np.matmul(beam_book[:, i], user.conj().T))
            if temp > gain:
                gain = temp
                index = i
        group_Frf.append(index)
    return group_Frf

def getFbb_classic(group, F_rf, Pw):
    H_bb = []
    for cl in group:
        h = np.sqrt(Pw) * np.matmul(F_rf.conjugate().T, cl)
        H_bb.append(h)

    F_bb = computeDigitalBf(np.asarray(H_bb))

    for i in range(len(F_bb[0])):
        F_bb[:, i] = F_bb[:, i] / LA.norm(np.matmul(F_rf, F_bb[:, i]))

    return F_bb

def computeSINR_classic(user_order, F_bb, F_rf, group, Pw, thegam):
    up = LA.norm(np.matmul(F_bb[:, user_order].conj().T, np.matmul(F_rf.conj().T, group[user_order]))) ** 2 * Pw
    down = 0
    for i in range(len(group)):
        if i == user_order: continue
        down += LA.norm(np.matmul(F_bb[:, user_order].conj().T, np.matmul(F_rf.conj().T, group[i]))) ** 2 * Pw

    down += LA.norm(np.matmul(F_bb[:, user_order].conj().T, F_rf.conj().T)) ** 2 * thegam

    return up / down


def dB2thegam(SNR):
    return 1 / np.power(10, np.asarray(SNR) / 10)


def computeSE(SNR,  F_bb, F_rf, group, Pw):
    SE = []
    for thegam in dB2thegam(SNR):
        SINR_group = []
        for i in range(len(F_bb)):
            SINR_I = computeSINR_classic(i, F_bb, F_rf, group[:len(F_bb)], Pw, thegam)
            SINR_group.append(SINR_I)
        R_group = np.log2(1 + np.asarray(SINR_group))
        R = sum(R_group)
        SE.append(R)
    return SE



def demo():
    N_bs = N_beam = 128
    N_rf = 12
    user_num = 16
    Pw = 24

    beam_book = np.asarray([computeBfVector(i / N_beam * 2 * np.pi) for i in range(N_beam)]).T
    group = np.asarray([computeChannel() for i in range(user_num)])
    group_frf = beamSelect(group=group[:N_rf], beam_book=beam_book)
    F_rf = beam_book[:, group_frf]
    F_bb = getFbb_classic(group[:N_rf], F_rf, Pw)

    SNR = np.linspace(0, 20, 12)

    SE = computeSE(SNR, F_bb, F_rf, group, Pw)


    plt.plot(SNR, SE)
    plt.xlabel("dB")
    plt.ylabel("SE")

    plt.show()
    pass




if __name__ == "__main__":
    demo()
