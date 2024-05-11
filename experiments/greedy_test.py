import pandas as pd

from utils import compute_significance

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset, coverage, diversity
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.rl import GroupExplorationEnv, TrueOnlineSarsaLambda
from hypothesis_exploration.greedy_algorithm import GreedyExplorer

from datasets.MovieLens import params as movie_params
from datasets.BookCrossing import params as book_params
from datasets.Yelp import params as yelp_params

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

movie_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=4, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='mean', null_value=3.5, alternative='less', n_sample=HypothesisTest.ONE_SAMPLE),
]

eta = 1
alpha = 0.05
gamma = 500
lambd = 1
n = 3
m = 35
initial_wealth = eta * alpha

datasets = {
    'MovieLens': (movie_dataset, movie_hypotheses),
}

for dataset_name, dataset_variables in datasets.items():

    dataset = dataset_variables[0]
    hypotheses = dataset_variables[1]

    # Greedy-HEP
    greedy = GreedyExplorer(
        D=dataset,
        H=hypotheses,
        alpha=alpha,
        n=n,
        eta=eta,
        gamma=gamma,
        lambd=lambd,
        starting_predicates={'genre': 'Comedy'}
    )

    greedy.reset()

    for step in range(m):
        selected_g_in, selected_G_out, selected_h = greedy.step()
        print(selected_h)
        print(str(selected_g_in) + ' - ' + str(len(selected_g_in.user_ids)) + 'users')
        for g in selected_G_out:
            print('\t' + str(g) + ' - ' + str(len(g.user_ids)) + 'users')
        print()

        cov = coverage(selected_G_out, selected_g_in)
        div = diversity(selected_G_out, normalized=False)
        div_norm = diversity(selected_G_out, normalized=True)