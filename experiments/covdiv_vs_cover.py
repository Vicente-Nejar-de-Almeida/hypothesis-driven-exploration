import random
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset, Group
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha, cover_alpha

from datasets.MovieLens import params


dataframe = pd.read_csv('../datasets/MovieLens/MovieLens.csv')

dataset = Dataset(dataframe=dataframe, multi_value_attribute_names=params.multi_value_attribute_names, attributes=params.attributes, action_dimension=params.action_dimension)


def select_random_group(num_of_attributes):
    random_predicates = {}
    random_attributes = random.sample(list(params.attributes), num_of_attributes)
    for att in random_attributes:
        random_predicates[att] = random.choice(params.attributes[att])
    return Group(dataset=dataset, predicates=random_predicates)


hypothesis = HypothesisTest(aggregation='mean', null_value=3, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

eta = 1
alpha = 0.05
lambd = 500
# n = 4
initial_wealth = eta * alpha
w1 = 0.5
w2 = 0.5

runs = 3
groups_per_run = 2

coverage_results = {
    'n': [],
    'COVDIV_alpha': [],
    'COVER_alpha': [],
}

diversity_results = {
    'n': [],
    'COVDIV_alpha': [],
    'COVER_alpha': [],
}

objective_results = {
    'n': [],
    'COVDIV_alpha': [],
    'COVER_alpha': [],
}

for run in range(runs):
    print(f'Run {run+1}')
    for _ in range(groups_per_run):
        for n in [2, 4, 6, 8, 10]:
            g_in = select_random_group(1)
            coverage_results['n'].append(n)
            diversity_results['n'].append(n)
            objective_results['n'].append(n)
            covdiv_G_out, covdiv_current_wealth, covdiv_obj_value, covdiv_cov_value, covdiv_div_value, covdiv_tested_requests = covdiv_alpha(D=dataset, g_in = g_in, h=hypothesis, alpha=alpha, n = 4, wealth=initial_wealth, lambd=lambd, w1=w1, w2=w2, request_history=[])
            coverage_results['COVDIV_alpha'].append(covdiv_cov_value)
            diversity_results['COVDIV_alpha'].append(covdiv_div_value)
            objective_results['COVDIV_alpha'].append(covdiv_obj_value)
            cover_G_out, cover_current_wealth, cover_cov_value, cover_div_value, cover_tested_requests = cover_alpha(D=dataset, g_in = g_in, h=hypothesis, alpha=alpha, n = 4, wealth=initial_wealth, lambd=lambd)
            coverage_results['COVER_alpha'].append(cover_cov_value)
            diversity_results['COVER_alpha'].append(cover_div_value)
            objective_results['COVER_alpha'].append(w1 * cover_cov_value + w2 * cover_div_value)

coverage_results = pd.DataFrame(coverage_results).groupby('n').mean()
diversity_results = pd.DataFrame(diversity_results).groupby('n').mean()
objective_results = pd.DataFrame(objective_results).groupby('n').mean()

coverage_results.plot.bar(title='Coverage')
plt.show()

diversity_results.plot.bar(title='Diversity')
plt.show()

objective_results.plot.bar(title='Objective function (w1 = 0.5, w2 = 0.5)')
plt.show()
