import gym

from math import floor
from gym import spaces
from hypothesis_exploration.user_data_model.data_model import Dataset, Group
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest
from hypothesis_exploration.alpha_investing.covdiv_alpha import covdiv_alpha

class GroupExplorationEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, D: Dataset, H: [HypothesisTest], alpha: float, n: float, eta: float, lambd: float, w1: float, w2: float):
		super(GroupExplorationEnv, self).__init__()
		self.D = D
		self.H = H
		self.alpha = alpha
		self.n = n
		self.eta = eta
		self.lambd = lambd
		self.w1 = w1
		self.w2 = w2

		"""
		Available actions are selecting a pair (g_i, h_j) to give to covdiv_alpha.
		i = floor(action / len(H))
		j = action % len(H)
		"""
		self.action_space = spaces.Discrete(self.n * len(self.H))

		self.observation_space = spaces.Dict({
		f'g{i+1}': spaces.Box(0, 1, shape=(self.D.group_encoded_len,), dtype=float) for i in range(self.n)
		})

		empty_dataframe = self.D.dataframe.head(0)
		empty_dataset = Dataset(
			dataframe=empty_dataframe,
			attributes=self.D.attributes,
			multi_value_attribute_names=self.D.multi_value_attribute_names,
			action_dimension=self.D.action_dimension,
		)
		self._empty_group = Group(dataset=empty_dataset, predicates={})

	def _get_obs(self):
		return {f'g{i+1}': self._groups[i].encode() for i in range(self.n)}

	def step(self, action):
		i = floor(action / len(self.H))
		j = action % len(self.H)

		g_in = self._groups[i]
		h = self.H[j]
		
		G_out, self._wealth, obj_value, cov_value, div_value, tested_requests = covdiv_alpha(D=self.D, g_in=g_in, h=h, alpha=self.alpha, n=self.n, wealth=self._wealth, lambd=self.lambd, w1=self.w1, w2=self.w2, request_history=[])
		self._groups = list(G_out)
		self._groups += [self._empty_group for _ in range(self.n - len(G_out))]
		
		observation = self._get_obs()
		reward = obj_value
		done = (len(G_out) == 0)
		info = {
			'g_in': str(g_in),
			'h': str(h),
			'G_out': [str(g) for g in G_out],
			'wealth': self._wealth,
		}
		return observation, reward, done, info

	def reset(self, seed=None):
		super().reset(seed=seed)
		self._wealth = self.eta * self.alpha
		self._groups = [Group(dataset=self.D, predicates={})]
		self._groups += [self._empty_group for _ in range(self.n - 1)]
		observation = self._get_obs()
		return observation  # reward, done, info can't be included

	def render(self, mode='human'):
		pass

	def close(self):
		pass
