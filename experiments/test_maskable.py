import pandas as pd

from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.rl import GroupExplorationEnv

from datasets.MovieLens import params

dataframe = pd.read_csv('../datasets/MovieLens/MovieLens.csv')

dataset = Dataset(
    dataframe=dataframe,
    multi_value_attribute_names=params.multi_value_attribute_names,
    attributes=params.attributes,
    action_dimension=params.action_dimension,
    action_dimension_min=params.action_dimension_min,
    action_dimension_max=params.action_dimension_max
)

hypothesis = HypothesisTest(aggregation='mean', null_value=3, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

eta = 1
alpha = 0.05
lambd = 500
n = 4
initial_wealth = eta * alpha
w1 = 0.5
w2 = 0.5


def make_env():
    return GroupExplorationEnv(
        D=dataset,
        H=[hypothesis],
        alpha=alpha,
        n=n,
        eta=eta,
        lambd=lambd,
        w1=w1,
        w2=w2,
    )


env = DummyVecEnv([make_env])
model = MaskablePPO("MultiInputPolicy", env, gamma=1, seed=32, verbose=1, n_steps=10)
model.learn(total_timesteps=30, log_interval=1)

# model.save("hypo_explorer")
# del model
# model = MaskablePPO.load("hypo_explorer")

env = make_env()
obs, info = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
    print(info)
    done = terminated or truncated
