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
    HypothesisTest(aggregation='mean', null_value=3.4, alternative='less', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='variance', null_value=1.4, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
]


movie_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=4.9, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='mean', null_value=3.5, alternative='less', n_sample=HypothesisTest.ONE_SAMPLE),
    HypothesisTest(aggregation='variance', null_value=4.4, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
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

hypothesis_to_color = {
    0: '#636EFA',
    1: '#EF553B',
    2: '#00CC96',
    3: '#AB63FA',
    4: '#FFA15A',
    5: '#19D3F3',
    6: '#FF6692',
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

    ids = []
    labels = []
    color_sequence = []
    parents = []
    user_counts = []

    for step in range(m):
        selected_g_in, selected_G_out, selected_h = greedy.step()
        print(selected_h)
        print(selected_g_in.to_string_in_exploration_order() + ' - ' + str(len(selected_g_in.user_ids)) + 'users')

        
        if len(parents) == 0:
            if len(selected_g_in.predicates) > 0:
                ids.append(','.join(selected_g_in.predicates.values()))
                labels.append(list(selected_g_in.predicates.values())[-1])
            else:
                ids.append('')
                labels.append(dataset_name)
            color_sequence.append('')
            parents.append('')
            user_counts.append(len(selected_g_in.user_ids))
            
        
        for g in selected_G_out:
            print('\t' + g.to_string_in_exploration_order() + ' - ' + str(len(g.user_ids)) + 'users')
            ids.append(','.join(g.predicates.values()))
            labels.append(list(g.predicates.values())[-1])
            color_sequence.append(hypothesis_to_color[hypotheses.index(selected_h)])
            parents.append(','.join(selected_g_in.predicates.values()))
            user_counts.append(len(g.user_ids))
        print()

        cov = coverage(selected_G_out, selected_g_in)
        div = diversity(selected_G_out, normalized=False)
        div_norm = diversity(selected_G_out, normalized=True)
    
    print(f'ids = {ids}')
    print(f'labels = {labels}')
    print(f'color_sequence = {color_sequence}')
    print(f'parents = {parents}')
    print(f'user_counts = {user_counts}')
