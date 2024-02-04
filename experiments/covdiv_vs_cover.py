import random
import time
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from statsmodels.stats.multitest import multipletests

from hypothesis_exploration.user_data_model import Dataset, Group
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha, cover_alpha

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

def select_random_group(num_of_attributes, min_user_count=50):
    group = None
    while group is None:
        random_predicates = {}
        random_attributes = random.sample(list(params.attributes), num_of_attributes)
        for att in random_attributes:
            random_predicates[att] = random.choice(params.attributes[att])
        potential_group = Group(dataset=dataset, predicates=random_predicates)
        if len(potential_group.user_ids) > min_user_count:
            group = potential_group
    return Group(dataset=dataset, predicates=random_predicates)


hypothesis = HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

eta = 1
alpha = 0.05
lambd = 500
n_values = [2, 4, 6, 8, 10]
initial_wealth = eta * alpha
weight_values = [(1, 0), (0, 1), (0.25, 0.75), (0.75, 0.25), (0.5, 0.5)]

runs = 30

results = {
    'run': [],
    'n': [],
    'w1': [],
    'w2': [],
    'algorithm': [],
    'g_in': [],
    'G_out': [],
    'coverage': [],
    'diversity': [],
    'objective': [],
    'power': [],
    'fdr': [],
    'time': []
}

for run in range(runs):
    print(f'Run {run+1}')

    # g_in = select_random_group(random.choice([1, 2, 3]))
    g_in = select_random_group(random.choice([1, 2]))

    print('g_in:', g_in)
    
    for n in n_values:
        print('n:', n)

        for weights in weight_values:
            w1, w2 = weights
            print('w:', weights)

            t0_covdiv = time.time()
            covdiv_G_out, covdiv_current_wealth, covdiv_obj_value, covdiv_cov_value, covdiv_div_value, covdiv_tested_requests, covdiv_rejects, covdiv_pvals = covdiv_alpha(D=dataset, g_in = g_in, h=hypothesis, alpha=alpha, n = 4, wealth=initial_wealth, lambd=lambd, w1=w1, w2=w2, request_history=[])
            t1_covdiv = time.time()
            time_covdiv = t1_covdiv - t0_covdiv
            ground_truth_reject, ground_truth_pvals, _, ground_truth_alphacBonf = multipletests(covdiv_pvals, alpha=alpha, method='bonferroni')
            num_rejects_ground_truth = sum([1 if reject else 0 for reject in ground_truth_reject])
            num_rejects_covdiv = sum([1 if reject else 0 for reject in covdiv_rejects])
            true_positives_covdiv = sum([1 if reject_covdiv and reject_ground_truth else 0 for reject_covdiv, reject_ground_truth in zip(covdiv_rejects, ground_truth_reject)])
            false_positives_covdiv = sum([1 if reject_covdiv and not reject_ground_truth else 0 for reject_covdiv, reject_ground_truth in zip(covdiv_rejects, ground_truth_reject)])

            results['run'].append(run+1)
            results['n'].append(n)
            results['w1'].append(w1)
            results['w2'].append(w2)
            results['algorithm'].append('COVDIV_alpha')
            results['g_in'].append(str(g_in))
            results['G_out'].append('; '.join([str(g) for g in covdiv_G_out]))
            results['coverage'].append(covdiv_cov_value)
            results['diversity'].append(covdiv_div_value)
            results['objective'].append(covdiv_obj_value)

            if num_rejects_ground_truth > 0:
                results['power'].append(true_positives_covdiv / num_rejects_ground_truth)
            else:
                results['power'].append(np.nan)
            
            if num_rejects_covdiv > 0:
                results['fdr'].append(false_positives_covdiv / num_rejects_covdiv)
            else:
                results['fdr'].append(np.nan)

            results['time'].append(time_covdiv)

            t0_cover = time.time()
            cover_G_out, cover_current_wealth, cover_cov_value, cover_div_value, cover_tested_requests, cover_rejects, cover_pvals = cover_alpha(D=dataset, g_in = g_in, h=hypothesis, alpha=alpha, n = 4, wealth=initial_wealth, lambd=lambd)
            t1_cover = time.time()
            time_cover = t1_cover - t0_cover
            ground_truth_reject, ground_truth_pvals, _, ground_truth_alphacBonf = multipletests(cover_pvals, alpha=alpha, method='bonferroni')
            num_rejects_ground_truth = sum([1 if reject else 0 for reject in ground_truth_reject])
            num_rejects_cover = sum([1 if reject else 0 for reject in cover_rejects])
            true_positives_cover = sum([1 if reject_cover and reject_ground_truth else 0 for reject_cover, reject_ground_truth in zip(cover_rejects, ground_truth_reject)])
            false_positives_cover = sum([1 if reject_cover and not reject_ground_truth else 0 for reject_cover, reject_ground_truth in zip(cover_rejects, ground_truth_reject)])

            results['run'].append(run+1)
            results['n'].append(n)
            results['w1'].append(w1)
            results['w2'].append(w2)
            results['algorithm'].append('COVER_alpha')
            results['g_in'].append(str(g_in))
            results['G_out'].append('; '.join([str(g) for g in cover_G_out]))
            results['coverage'].append(cover_cov_value)
            results['diversity'].append(cover_div_value)
            results['objective'].append(w1 * cover_cov_value + w2 * cover_div_value)

            if num_rejects_ground_truth > 0:
                results['power'].append(true_positives_cover / num_rejects_ground_truth)
            else:
                results['power'].append(np.nan)
            
            if num_rejects_cover > 0:
                results['fdr'].append(false_positives_cover / num_rejects_cover)
            else:
                results['fdr'].append(np.nan)
            
            results['time'].append(time_cover)

results = pd.DataFrame(results)
results.to_csv('../results/MovieLens/covdiv_vs_cover.csv', index=False)
