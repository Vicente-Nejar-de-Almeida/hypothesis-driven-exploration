import time
import numpy as np
import pandas as pd

from utils import compute_significance

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset, Group, coverage, diversity
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha, cover_alpha

from datasets.MovieLens import params as movie_params
from datasets.BookCrossing import params as book_params
from datasets.Yelp import params as yelp_params


def run_alpha_investing(algorithm, has_lambd, **kwargs):
    if not has_lambd:
        if 'lambd' in kwargs:
            del kwargs['lambd']

    request_history = {}

    t0 = time.time()
    G_out, wealth = algorithm(**kwargs, request_history=request_history)
    t1 = time.time()

    execution_time = t1 - t0
    cov = coverage(G_out, kwargs['g_in'])
    div = diversity(G_out, normalized=False)

    power, fdr = compute_significance(request_history, kwargs['alpha'])

    return G_out, wealth, cov, div, power, fdr, execution_time


movie_dataframe = pd.read_csv('../datasets/MovieLens/MovieLens.csv')
book_dataframe = pd.read_csv('../datasets/BookCrossing/BookCrossing.csv')
yelp_dataframe = pd.read_csv('../datasets/Yelp/Yelp.csv')

movie_dataset = Dataset(
    dataframe=movie_dataframe,
    multi_value_attribute_names=movie_params.multi_value_attribute_names,
    attributes=movie_params.attributes,
    action_dimension=movie_params.action_dimension,
    action_dimension_min=movie_params.action_dimension_min,
    action_dimension_max=movie_params.action_dimension_max
)

book_dataset = Dataset(
    dataframe=book_dataframe,
    multi_value_attribute_names=book_params.multi_value_attribute_names,
    attributes=book_params.attributes,
    action_dimension=book_params.action_dimension,
    action_dimension_min=book_params.action_dimension_min,
    action_dimension_max=book_params.action_dimension_max
)

yelp_dataset = Dataset(
    dataframe=yelp_dataframe,
    multi_value_attribute_names=yelp_params.multi_value_attribute_names,
    attributes=yelp_params.attributes,
    action_dimension=yelp_params.action_dimension,
    action_dimension_min=yelp_params.action_dimension_min,
    action_dimension_max=yelp_params.action_dimension_max
)

movie_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='variance', null_value=1, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='distribution', null_value='norm', alternative='two-sided', n_sample=HypothesisTest.ONE_SAMPLE),
]

movie_input_groups = [
    Group(dataset=movie_dataset, predicates={'age': '18-24'}),
    Group(dataset=movie_dataset, predicates={'gender': 'F'}),
    Group(dataset=movie_dataset, predicates={'genre': 'Sci-Fi'}),
    Group(dataset=movie_dataset, predicates={'occupation': 'programmer'}),
    Group(dataset=movie_dataset, predicates={'runtime_minutes': 'Very Long'}),
]

book_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=2, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='variance', null_value=4, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='distribution', null_value='norm', alternative='two-sided', n_sample=HypothesisTest.ONE_SAMPLE),
]

book_input_groups = [
    Group(dataset=book_dataset, predicates={'age': '18-24'}),
    Group(dataset=book_dataset, predicates={'category': 'Fiction'}),
    Group(dataset=book_dataset, predicates={'country': 'usa'}),
    Group(dataset=book_dataset, predicates={'language': 'en'}),
    Group(dataset=book_dataset, predicates={'year_of_publication': '90s'}),
]

yelp_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='variance', null_value=1, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='distribution', null_value='norm', alternative='two-sided', n_sample=HypothesisTest.ONE_SAMPLE),
]

yelp_input_groups = [
    Group(dataset=yelp_dataset, predicates={'category': 'Automotive'}),
    Group(dataset=yelp_dataset, predicates={'category': 'Health & Medical'}),
    Group(dataset=yelp_dataset, predicates={'city': 'Montréal'}),
    Group(dataset=yelp_dataset, predicates={'city': 'Toronto'}),
    Group(dataset=yelp_dataset, predicates={'fans': 'popular'}),
]


eta = 1
alpha = 0.05
gamma = 500
lambd = 1
num_output_groups = [i for i in range(1, 21)]
initial_wealth = eta * alpha

datasets = {
    'MovieLens': (movie_dataset, movie_input_groups, movie_hypotheses),
    'BookCrossing': (book_dataset, book_input_groups, book_hypotheses),
    'Yelp': (yelp_dataset, yelp_input_groups, yelp_hypotheses),
}

algorithms = {
    'covdiv_alpha': covdiv_alpha,
    'cover_alpha': cover_alpha,
}

for dataset_name, dataset_variables in datasets.items():
    print(dataset_name)

    dataset = dataset_variables[0]
    input_groups = dataset_variables[1]
    hypotheses = dataset_variables[2]

    results = {
        'algorithm': [],
        'n': [],
        'g_in': [],
        'h': [],
        'G_out': [],
        'coverage': [],
        'diversity': [],
        'power': [],
        'fdr': [],
        'time': []
    }

    for algorithm_name, algorithm in algorithms.items():
        print('algorithm', algorithm_name)
        for g_in in input_groups:
            print('\tg_in:', g_in)
            for h in hypotheses:
                print('\t\th:', h)
                for n in num_output_groups:
                    print('\t\t\tn', n)
                    G_out, wealth, cov, div, power, fdr, execution_time = run_alpha_investing(
                        algorithm=algorithm,
                        has_lambd=(algorithm_name == 'covdiv_alpha'),
                        D=dataset,
                        g_in=g_in,
                        h=h,
                        alpha=alpha,
                        n=n,
                        wealth=initial_wealth,
                        gamma=gamma,
                        lambd=lambd,
                    )
                    results['algorithm'].append(algorithm_name)
                    results['n'].append(n)
                    results['g_in'].append(g_in)
                    results['h'].append(str(h))
                    results['G_out'].append(', '.join([str(g) for g in G_out]))
                    results['coverage'].append(cov)
                    results['diversity'].append(div)
                    results['power'].append(power)
                    results['fdr'].append(fdr)
                    results['time'].append(execution_time)

    results = pd.DataFrame(results)
    results.to_csv(f'../results/{dataset_name}/fixing_result.csv', index=False)
