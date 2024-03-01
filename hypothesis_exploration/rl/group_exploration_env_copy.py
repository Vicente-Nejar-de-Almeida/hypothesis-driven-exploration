import random
import gymnasium as gym
import numpy as np

from queue import PriorityQueue
from math import floor
from gymnasium import spaces
from hypothesis_exploration.user_data_model.data_model import Dataset, Group, coverage, diversity
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha, cover_alpha


class GroupExplorationEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, D: Dataset, H: list[HypothesisTest], alpha: float, n: float, m: int, eta: float, gamma: float, lambd: float, additional_group_count: int, max_predicates: int):
    # def __init__(self, D: Dataset, H: list[HypothesisTest], alpha: float, n: float, m: int, eta: float, gamma: float, lambd: float, max_predicates: int):
        super(GroupExplorationEnv, self).__init__()
        self.D = D
        self.H = H
        self.alpha = alpha
        self.n = n
        self.m = m
        self.eta = eta
        self.gamma = gamma
        self.lambd = lambd
        self.additional_group_count = additional_group_count
        self.max_predicates = max_predicates
        self.previously_generated_groups = {}

        self.possible_attributes = {att: [] for att in self.D.attributes.keys()}
        for att in self.D.attributes:
            self.possible_attributes[att] += list(self.D.dataframe[att].value_counts()[:5].index)

        # self.q = PriorityQueue()

        """
        Available actions are selecting a pair (g_i, h_j) to give to covdiv_alpha.
        i = floor(action / len(H))
        j = action % len(H)
        """
        
        self.action_space = spaces.Discrete((self.n + self.additional_group_count) * len(self.H))
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
            low=np.zeros((self.n + self.additional_group_count) * self.D.group_encoded_len, dtype=np.float32),
            high=np.ones((self.n + self.additional_group_count) * self.D.group_encoded_len, dtype=np.float32),
        )
    
    def select_random_group(self):
        random_predicate = {}
        random_attribute = random.choice(list(self.possible_attributes))
        random_attribute_value = random.choice(self.D.attributes[random_attribute])
        random_predicate[random_attribute] = random_attribute_value
        group = Group(dataset=self.D, predicates=random_predicate)
        if (random_attribute, random_attribute_value) in self.previously_generated_groups:
            group = self.previously_generated_groups[(random_attribute, random_attribute_value)]
        else:
            group = Group(dataset=self.D, predicates=random_predicate)
            self.previously_generated_groups[(random_attribute, random_attribute_value)] = group
        return group

    def _get_obs(self):
        
        groups = []
        for i in range(self.n + self.additional_group_count):
            groups += list(self._groups[i].encoded)
        return np.array(groups, dtype=np.float32)
        
        # return {f'g{i+1}': self._groups[i].encoded for i in range(self.n + self.additional_group_count)}
        
        # return {f'g{i+1}': self._groups[i].encoded for i in range(self.n)}

    def step(self, action):
        self._step += 1
        # print(f'Action: {action}, Groups: {[str(g) for g in self._groups]}')
        i = floor(action / len(self.H))
        j = action % len(self.H)

        g_in = self._groups[i]
        h = self.H[j]

        """
        for l, g in zip(self._group_levels, self._groups):
            if (g != g_in) and (l is not None):
                self.q.put((l, g))
        """

        # print('Before covdiv')
        
        G_out, self._wealth = covdiv_alpha(D=self.D, g_in=g_in, h=h, alpha=self.alpha, n=self.n, wealth=self._wealth, gamma=self.gamma, lambd=self.lambd, request_history=self._request_history)

        # print('After covdiv')

        x = (1 / (1 + self.lambd))
        # obj_value = x * coverage(G_out, g_in) + x * self.lambd * diversity(G_out, normalized=True)
        obj_value = coverage(G_out, g_in) + self.lambd * diversity(G_out)

        if (len(G_out) > 0) and (len(list(G_out)[0].predicates) < self.max_predicates):
            self._groups = list(G_out)
        else:
            self._groups = []
        
        """
        current_level = self._group_levels[i]
        if (len(G_out) > 0) and (len(list(G_out)[0].predicates) < self.max_predicates):
            self._groups = list(G_out)
            self._group_levels = [current_level for _ in range(len(G_out))]
        else:
            self._groups = []
            self._group_levels = []

        for _ in range(self.n + self.additional_group_count - len(self._groups)):
            if not self.q.empty():
                level, group = self.q.get()
            else:
                level, group = None, self._empty_group
            self._groups.append(group)
            self._group_levels.append(level)
        """

        for _ in range(self.n + self.additional_group_count - len(self._groups)):
            self._groups.append(self.select_random_group())
        
        observation = self._get_obs()
        reward = obj_value
        # done = (self._step == self.m) or (self._wealth <= 0) or (self._group_levels == [None] * len(self._group_levels))
        done = (self._step == self.m) or (self._wealth <= 0)
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
        self._groups = [self.select_random_group() for _ in range(self.n + self.additional_group_count)]
        # self._groups = [Group(dataset=self.D, predicates={})]
        # self._groups += [select_random_group(self.D) for _ in range(self.n - 1)]
        # self._groups += [self._empty_group for _ in range(self.n + self.additional_group_count - 1)]
        # self._group_levels = [0] + [None for _ in range(self.n + self.additional_group_count - 1)]
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
        return mask
