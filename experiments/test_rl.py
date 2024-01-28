import gymnasium as gym
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

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

env = GroupExplorationEnv(
    D=dataset,
    H=[hypothesis],
    alpha=alpha,
    n=n,
    eta=eta,
    lambd=lambd,
    w1=w1,
    w2=w2,
)

tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = DQN("MultiInputPolicy", env, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=10, log_interval=4)

# model.save("hypo_explorer")
# del model
# model = DQN.load("hypo_explorer")

obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(info)
    done = terminated or truncated
