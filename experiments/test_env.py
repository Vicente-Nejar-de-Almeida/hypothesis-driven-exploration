import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.rl import GroupExplorationEnv

from datasets.MovieLens import params

dataframe = pd.read_csv('../datasets/MovieLens/MovieLens.csv')

dataset = Dataset(dataframe=dataframe, multi_value_attribute_names=params.multi_value_attribute_names, attributes=params.attributes, action_dimension=params.action_dimension)

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

observation = env.reset()
done = False
while not done:
    action = 0
    observation, reward, done, info = env.step(action=action)
    print(info)
