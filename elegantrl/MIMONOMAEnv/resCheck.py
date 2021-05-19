import torch
import numpy as np
import matplotlib.pyplot as plt
from net import ActorPPO

from EnvScript import MIMONOMAEnv

def get_episode_return_and_act(env, act, step, device):
    episode_returns = []
    act_list = []
    ereturn = 0.0  # sum of rewards in an episode
    max_step = step

    state = env.reset()
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
    return act_list, episode_returns


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数


if __name__ == "__main__":
    path = "../AgentPPO/20210516_MIMONOMAEnv-v0_0"
    step = 2 ** 10
    window = 20

    env = MIMONOMAEnv()
    actor = ActorPPO(2 ** 9, env.state_dim, env.action_dim).to("cpu")
    actor.load_state_dict(torch.load(path + "/actor.pth"))

    act_list, episode_return = get_episode_return_and_act(env, actor, step, "cpu")
    act_list = np.asarray(act_list)
    episode_return = np.asarray(episode_return)
    episode_return = moving_average(episode_return, window)

    plt.subplot(2, 1, 1)
    plt.hist(act_list[:, 0], bins=20, facecolor="royalblue", edgecolor="red")
    # 显示横轴标签
    plt.xlabel("action_theta")
    # 显示纵轴标签
    plt.ylabel("frequency")
    # 显示图标题
    plt.title("theta-frequency")

    plt.subplot(2, 1, 2)
    plt.hist(act_list[:, 1], bins=20, facecolor="royalblue", edgecolor="red")
    # 显示横轴标签
    plt.xlabel("action_power")
    # 显示纵轴标签
    plt.ylabel("frequency")
    # 显示图标题
    plt.title("power-frequency")
    plt.show()
    # plt.savefig(path + "/action_dis.jpg")


    plt.plot(episode_return, color="royalblue")
    plt.ylabel("episode_reward")
    plt.xlabel("episodes")
    plt.show()






