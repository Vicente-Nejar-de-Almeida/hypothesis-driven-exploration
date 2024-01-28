import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset, Group, generate_candidates
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha
from hypothesis_exploration.greedy_algorithm import GreedyExplorer

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

hypothesis = HypothesisTest(aggregation='mean', null_value=4.2, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

eta = 1
alpha = 0.05
lambd = 500
n = 4
w1 = 0.5
w2 = 0.5

greedy = GreedyExplorer(
    D=dataset,
    H=[hypothesis],
    alpha=alpha,
    n=n,
    eta=eta,
    lambd=lambd,
    w1=w1,
    w2=w2,
)
greedy.reset()

print('Initial wealth:', eta * alpha)

steps = 50
for step in range(steps):
    selected_g_in, selected_G_out = greedy.step()
    print(f'Step {step}, wealth: {greedy.wealth}')
    print(selected_g_in)
    for g in selected_G_out:
        print('\t' + str(g))
