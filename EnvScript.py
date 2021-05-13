from utils import userGrouping, transformState, latterPartState, \
    computeChABFGain, computeBfVector, computeSINR
from settingup import args

import numpy as np
import numpy.linalg as LA

class MIMONOMAEnv:

    def __init__(self,
                 groups_number=args.N_rf,
                 desired_user_number=args.desired_user_number,
                 N_bs=args.N_bs,
                 random_seed=args.seed):
        self.env_name = 'MIMONOMAEnv-v0'
        self.if_discrete = False
        self.target_reward = None
        self.max_step = desired_user_number
        self.seed = random_seed

        self.desired_user_number = desired_user_number
        self.groups_number = groups_number
        self.user_number = None
        self.N_bs = N_bs

        self.state_dim = (3 * 2, self.desired_user_number, self.N_bs)
        self.action_dim = 2

        self.state = None
        self.groups = None

        self.group_index = None
        self.Pw = {}
        self.theta = {}

        self.order = 0

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

        self.theta[curr_group_index] = action[0]
        self.Pw[curr_group_index] = action[1]

        reward = computeChABFGain(self.groups[curr_group_index], computeBfVector(action[0]))

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
            reward = reward + final_reward

        # update env state
        self.state = state
        self.group_index = self.group_index + 1

        print(self.order, ": ", "reward:", reward, "done:", done)
        self.order += 1

        return state, reward, done, None

    def _currActGroupsIndex(self):
        index = 0
        while LA.norm(self.state[0][index]) == 0.0 and LA.norm(self.state[3][index]) == 0.0:
            index += 1
        return int(self.state[2][index][0])


if __name__ == '__main__':
    env = MIMONOMAEnv()
    state1 = env.reset()
    done = False
    while True:
        if done:
            env.reset()
        env.state, reward, done, ___ = env.step((1.5 * np.random.random(), 5 * np.random.random()))
