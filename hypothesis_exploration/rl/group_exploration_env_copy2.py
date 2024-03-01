import random
import gymnasium as gym
import numpy as np
import copy

from queue import PriorityQueue
from math import floor
from gymnasium import spaces
from hypothesis_exploration.user_data_model.data_model import Dataset, Group, coverage, diversity
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha, cover_alpha


class GroupExplorationEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, D: Dataset, H: list[HypothesisTest], alpha: float, n: float, m: int, eta: float, gamma: float, lambd: float, max_predicates: int):
        super(GroupExplorationEnv, self).__init__()
        self.D = D
        self.H = H
        self.alpha = alpha
        self.n = n
        self.m = m
        self.eta = eta
        self.gamma = gamma
        self.lambd = lambd
        self.max_predicates = max_predicates
        self.previously_generated_groups = {}
        self.previous_levels = []
        self.add_to_previous_levels = None

        # self.q = PriorityQueue()

        """
        Available actions are selecting a pair (g_i, h_j) to give to covdiv_alpha.
        i = floor(action / len(H))
        j = action % len(H)
        """
        
        self.action_space = spaces.Discrete((self.n) * len(self.H) + 1)
        # self.action_space = spaces.Discrete(self.n * len(self.H))

        """
        self.observation_space = spaces.Dict({
            f'g{i+1}': spaces.Box(0, 1, shape=(self.D.group_encoded_len,), dtype=np.float32) for i in range(self.n)
        })
        """

        """
        self.observation_space = spaces.Dict({
        f'g{i+1}': spaces.Box(0, 1, shape=(self.D.group_encoded_len,), dtype=np.float32) for i in range(self.n + self.additional_group_count)
        })
        """

        
        self.observation_space = spaces.Box(
            low=np.zeros((self.n) * self.D.group_encoded_len, dtype=np.float32),
            high=np.ones((self.n) * self.D.group_encoded_len, dtype=np.float32),
        )

        empty_dataframe = self.D.dataframe.head(0)
        empty_dataset = Dataset(
            dataframe=empty_dataframe,
            attributes=self.D.attributes,
            multi_value_attribute_names=self.D.multi_value_attribute_names,
            action_dimension=self.D.action_dimension,
            action_dimension_min=self.D.action_dimension_min,
            action_dimension_max=self.D.action_dimension_max,
        )
        self._empty_group = Group(dataset=empty_dataset, predicates={})

    def _get_obs(self):
        groups = []
        for i in range(self.n):
            groups += list(self._groups[i].encoded)
        return np.array(groups, dtype=np.float32)
        
        # return {f'g{i+1}': self._groups[i].encoded for i in range(self.n + self.additional_group_count)}
        
        # return {f'g{i+1}': self._groups[i].encoded for i in range(self.n)}

    def step(self, action):

        if action == (self.n * len(self.H)):
            reward = 0
            self.add_to_previous_levels = self.previous_levels.pop()
            self._groups = copy.deepcopy(self.add_to_previous_levels)
            # self._groups = self.previous_levels.pop()
            for _ in range(self.n - len(self._groups)):
                self._groups.append(self._empty_group)
            observation = self._get_obs()
            return observation, reward, False, False, {}

        i = floor(action / len(self.H))
        j = action % len(self.H)

        if self.add_to_previous_levels is not None:
            if len(self.add_to_previous_levels) > 1:
                self.previous_levels.append([g for index, g in enumerate(self.add_to_previous_levels) if index != i])
            self.add_to_previous_levels = None

        g_in = self._groups[i]
        h = self.H[j]
        
        G_out, self._wealth = covdiv_alpha(D=self.D, g_in=g_in, h=h, alpha=self.alpha, n=self.n, wealth=self._wealth, gamma=self.gamma, lambd=self.lambd, request_history=self._request_history)
        self._step += 1

        x = (1 / (1 + self.lambd))
        # obj_value = x * coverage(G_out, g_in) + x * self.lambd * diversity(G_out, normalized=True)
        obj_value = coverage(G_out, g_in) + self.lambd * diversity(G_out)

        if (len(G_out) > 0) and (len(list(G_out)[0].predicates) < self.max_predicates):
            self._groups = list(G_out)
            # self.add.append(list(G_out))
            self.add_to_previous_levels = list(G_out)
        else:
            self._groups = []
        
        reward = obj_value
        
        done = (self._step > self.m) or (self._wealth <= 0) or (len(self._groups) == 0 and len(self.previous_levels) == 0)
        
        for _ in range(self.n - len(self._groups)):
            self._groups.append(self._empty_group)
        observation = self._get_obs()

        info = {
            'g_in_str': str(g_in),
            'g_in': g_in,
            'h_str': str(h),
            'h': h,
            'G_out_str': [str(g) for g in G_out],
            'G_out': G_out,
            'wealth': self._wealth,
        }

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._wealth = self.eta * self.alpha
        self._groups = [Group(dataset=self.D, predicates={})]
        self._groups += [self._empty_group for _ in range(self.n - 1)]
        self._request_history = {}
        observation = self._get_obs()
        info = {}
        return observation, info  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def action_masks(self):
        mask = []
        for g in self._groups:
            if len(g.user_ids) > 0:
                mask += [True for _ in self.H]
            else:
                mask += [False for _ in self.H]
        if len(self.previous_levels) == 0:
            mask += [False]
        else:
            mask += [True]
        return mask
