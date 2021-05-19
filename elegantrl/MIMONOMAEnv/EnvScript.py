from utils import userGrouping, transformState, latterPartState, \
    computeChABFGain, computeBfVector, computeSINR
from settingup import args_channel

import numpy as np
import numpy.linalg as LA


class MIMONOMAEnv:

    def __init__(self,
                 groups_number=args_channel.N_rf,
                 desired_user_number=args_channel.desired_user_number,
                 N_bs=args_channel.N_bs,
                 random_seed=args_channel.seed,
                 P_min=args_channel.P_min, P_max=args_channel.P_max,
                 abf_discount=args_channel.abf_discount):
        self.env_name = 'MIMONOMAEnv-v0'
        self.if_discrete = False
        self.target_reward = None
        self.action_max = None
        self.max_step = desired_user_number
        self.seed = random_seed

        self.desired_user_number = desired_user_number
        self.groups_number = groups_number
        self.user_number = None
        self.N_bs = N_bs
        self.action_change = 0.3
        self.abf_discount = abf_discount[self.desired_user_number]

        self.state_dim = (6, self.desired_user_number, self.N_bs)
        self.action_dim = 2

        self.state = None
        self.groups = None

        self.group_index = None
        self.Pw = {}
        self.theta = {}

        self.P_min = P_min
        self.P_max = P_max
        # self.order = 0

    def _updateGroups(self):
        self.user_number = int(np.random.random() * self.desired_user_number)
        if self.user_number < self.groups_number:
            self.user_number = self.groups_number
        self.groups = userGrouping(groups=self.groups_number, user_numbers=self.user_number)

    def reset(self) -> np.ndarray:
        self._updateGroups()
        state_part1 = transformState(groups_dict=self.groups)
        state_part2 = latterPartState(state_part1)
        state_true = np.vstack((state_part1, state_part2))

        differ = self.desired_user_number - len(state_true[0])
        if differ < 0:
            print("state dimension error. ")
            assert False
        elif differ == 0:
            state = state_true
        else:
            state = np.zeros((3 * 2, differ, self.N_bs))
            state = np.concatenate([state, state_true], axis=1)
        # update env state
        self.state = state

        self.Pw = {}
        self.theta = {}
        self.group_index = 0

        return state

    def step(self, action) -> (np.ndarray, float, bool, None):
        next_state_part1 = self.state[3:]
        next_state_part2 = latterPartState(next_state_part1)
        state = np.vstack((next_state_part1, next_state_part2))
        curr_group_index = self.group_index

        # np.clip(action[0], 0, 2 * np.pi)
        action_0 = (action[0] + 1) / 2 * 2 * np.pi
        action_0 = self.actionNormal(action_0)
        action_1 = (action[1] + 1) / 2 * self.P_max

        self.theta[curr_group_index] = action_0
        self.Pw[curr_group_index] = np.clip(action_1, self.P_min, self.P_max)

        reward = computeChABFGain(self.groups[curr_group_index], computeBfVector(action[0]))
        reward = ((reward / self.abf_discount) - 0.5) * 2

        done = False
        # is_done = LA.norm(next_state_part2[0]) + LA.norm(next_state_part2[1])
        is_done = curr_group_index == self.groups_number - 1
        if is_done:
            done = True
            final_reward = 0
            F_rf = [computeBfVector(self.theta[i]) for i in self.theta]
            F_rf = np.asarray(F_rf).T
            for i in self.groups:
                for j in range(len(self.groups[i])):
                    final_reward = final_reward + computeSINR(i, j, F_rf=F_rf, Pw=self.Pw, groups_dict=self.groups)
            final_reward = np.clip(((final_reward / self.abf_discount) - 0.5) * 2 * 5, - 5.0, 5.0)
            # print("reward: ", reward, "final_reward", final_reward)
            reward = reward + final_reward

        # update env state
        self.state = state
        self.group_index = self.group_index + 1

        # print(self.order, ": ", "reward:", reward, "done:", done)
        # self.order += 1

        return state, reward, done, None

    def actionNormal(self, action_0):
        """
        avoid the same action_0
        """
        nSame = 0
        while True:
            for i in self.theta:
                if action_0 == self.theta[i]:
                    action_0_1 = action_0 + action_0 * self.action_change * np.random.random() \
                        if action_0 != 0 else action_0 + (action_0 + 1) * self.action_change * np.random.random()
                    if action_0_1 <= 2 * np.pi:
                        action_0 = action_0_1
                    else:
                        action_0 = action_0 - action_0 * self.action_change * np.random.random()
                else:
                    nSame += 1
            if nSame == len(self.theta):
                break
            else:
                nSame = 0
        return action_0

    def _currActGroupsIndex(self):
        index = 0
        while LA.norm(self.state[0][index]) == 0.0 and LA.norm(self.state[3][index]) == 0.0:
            index += 1
        return int(self.state[2][index][0])


if __name__ == '__main__':
    env = MIMONOMAEnv()
    state1 = env.reset()
    done = False
    import time
    while True:
        if done:
            env.reset()
        env.state, reward, done, ___ = \
            env.step(((np.random.random() - 0.5) * 2, (np.random.random() - 0.5) * 2))
        time.sleep(0.2)

