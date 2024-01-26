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

dataset = Dataset(dataframe=dataframe, multi_value_attribute_names=params.multi_value_attribute_names, attributes=params.attributes, action_dimension=params.action_dimension)

hypothesis = HypothesisTest(aggregation='mean', null_value=3, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

eta = 1
alpha = 0.05
lambd = 500
n = 4
initial_wealth = eta * alpha
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

steps = 10
for _ in range(steps):
    selected_g_in, selected_G_out = greedy.step()
    print(selected_g_in)
    for g in selected_G_out:
        print('\t' + str(g))

"""
G_out, current_wealth, obj_value, cov_value, div_value, tested_requests = covdiv_alpha(D=dataset, g_in = root_group, h=hypothesis, alpha=alpha, n = 4, wealth=initial_wealth, lambd=lambd, w1=w1, w2=w2, request_history=[])
print('Initial wealth:', initial_wealth)
print('Resulting wealth:', current_wealth)
print('Objective function value:', obj_value)
print('Coverage value:', cov_value)
print('Diversity value:', div_value)
print('Output groups:')
for g in G_out:
    print(g)
"""

"""
sample = [0.4, 0.6, 0.7, 0.8, 0.9, 1]

test = HypothesisTest(aggregation='mean', null_value=0.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

# Mean greater than 0.5
print(test.test(sample))
"""
