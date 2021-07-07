import torch
from net import ActorPPO
from classicMethods import *
from utils import computeSINR
from EnvScript import MIMONOMAEnv

def get_episode_act(env, act, step, device):
    episode_returns = []
    act_list = []
    ereturn = 0.0  # sum of rewards in an episode
    max_step = step

    state = env.reset()
    init_state = env.users
    groups = env.groups

    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
        a_tensor = act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        act_list.append(action)
        ereturn += reward
        if done:
            state = env.reset()
            episode_returns.append(ereturn)
            ereturn = 0.0
    return act_list, init_state, groups

def computeSE_NN(theta, groups, Pw, SNR):
    SE = []
    for thegam in dB2thegam(SNR):
        final_reward = 0
        F_rf = [computeBfVector(theta[i]) for i in range(len(theta))]
        F_rf = np.asarray(F_rf).T
        for i in groups:
            for j in range(len(groups[i])):
                final_reward = final_reward + computeSINR(i, j, F_rf=F_rf, Pw=Pw, groups_dict=groups, sigma=thegam)
        SE.append(final_reward * 20)

    return SE


if __name__ == "__main__":
    N_bs = N_beam = 128
    N_rf = 12
    user_num = 16
    Pw = 24

    SNR = np.linspace(0, 20, 12)

    path = "../AgentPPO/MIMONOMAEnv-v0_0"
    step = N_rf

    env = MIMONOMAEnv()
    actor = ActorPPO(2 ** 8, env.state_dim, env.action_dim).to("cpu")
    actor.load_state_dict(torch.load(path + "/actor.pth"))

    act_list, init_state, groups = get_episode_act(env, actor, step, "cpu")
    act_list = np.asarray(act_list)
    action_theta = np.arctan(act_list[:, 1] / act_list[:, 0]) * 2 + np.pi

    Pw ={}
    for i in range(len(act_list[:, 2])):
        Pw[i] = abs(act_list[i, 2])

    SE_NN = computeSE_NN(action_theta, groups, Pw, SNR)

    beam_book = np.asarray([computeBfVector(i / N_beam * 2 * np.pi) for i in range(N_beam)]).T
    group = init_state
    group_frf = beamSelect(group=group[:N_rf], beam_book=beam_book)
    F_rf = beam_book[:, group_frf]
    F_bb = getFbb_classic(group[:N_rf], F_rf, 24)

    SE = computeSE(SNR, F_bb, F_rf, group, 24)

    plt.subplot(1, 2, 1)
    plt.plot(SNR, SE_NN)
    plt.ylabel("SE_NN")
    plt.xlabel("dB")

    plt.subplot(1, 2, 2)
    plt.plot(SNR, SE)
    plt.ylabel("SE")
    plt.xlabel("dB")

    plt.show()
    pass
