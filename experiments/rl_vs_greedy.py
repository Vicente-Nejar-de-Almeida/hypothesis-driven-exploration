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
]

book_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=2, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
]

yelp_hypotheses = [
    HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
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
    'BookCrossing': (book_dataset, book_hypotheses),
    'Yelp': (yelp_dataset, yelp_hypotheses),
}

for dataset_name, dataset_variables in datasets.items():
    print(dataset_name)

    dataset = dataset_variables[0]
    hypotheses = dataset_variables[1]

    pipeline_results = {
        'algorithm': [],
        'power': [],
        'fdr': [],
    }

    stepwise_results = {
        'algorithm': [],
        'step': [],
        'g_in': [],
        'h': [],
        'G_out': [],
        'coverage': [],
        'diversity': [],
        'normalized_diversity': [],
    }

    # Greedy-HEP
    print('\tRunning Greedy-HEP')
    greedy = GreedyExplorer(
        D=dataset,
        H=hypotheses,
        alpha=alpha,
        n=n,
        eta=eta,
        gamma=gamma,
        lambd=lambd,
    )

    greedy.reset()

    for step in range(m):
        selected_g_in, selected_G_out, selected_h = greedy.step()

        cov = coverage(selected_G_out, selected_g_in)
        div = diversity(selected_G_out, normalized=False)
        div_norm = diversity(selected_G_out, normalized=True)

        stepwise_results['algorithm'].append('Greedy-HEP')
        stepwise_results['step'].append(step)
        stepwise_results['g_in'].append(str(selected_g_in))
        stepwise_results['h'].append(str(selected_h))
        stepwise_results['G_out'].append(', '.join([str(g) for g in selected_G_out]))
        stepwise_results['coverage'].append(cov)
        stepwise_results['diversity'].append(div)
        stepwise_results['normalized_diversity'].append(div_norm)
    
    greedy_power, greedy_fdr = compute_significance(greedy.request_history, alpha)
    pipeline_results['algorithm'].append('Greedy-HEP')
    pipeline_results['power'].append(greedy_power)
    pipeline_results['fdr'].append(greedy_fdr)

    # RL-HEP
    print('\tTraining RL-HEP')
    env = GroupExplorationEnv(
        D=dataset,
        H=hypotheses,
        alpha=alpha,
        n=n,
        m=m,
        eta=eta,
        gamma=gamma,
        lambd=lambd,
        additional_group_count=1,
        max_predicates=4,
    )

    obs, info = env.reset()
    agent = TrueOnlineSarsaLambda(
        env.observation_space,
        env.action_space,
        alpha=0.001,
        fourier_order=5,
        gamma=1,
        lamb=0.9,
        epsilon=0.0,
        min_max_norm=True
    )

    episodes = 100
    for episode in range(episodes):
        print('\t\tEpisode', episode)
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            step += 1
    
    print('\tRunning RL-HEP')
    obs, info = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        cov = info['coverage']
        div = info['diversity']
        div_norm = diversity(info['G_out'], normalized=True)

        stepwise_results['algorithm'].append('RL-HEP')
        stepwise_results['step'].append(step)
        stepwise_results['g_in'].append(str(info['g_in']))
        stepwise_results['h'].append(str(info['h']))
        stepwise_results['G_out'].append(', '.join([str(g) for g in info['G_out']]))
        stepwise_results['coverage'].append(cov)
        stepwise_results['diversity'].append(div)
        stepwise_results['normalized_diversity'].append(div_norm)

        done = terminated or truncated
        # agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs
        step += 1
    
    rl_power, rl_fdr = compute_significance(env._request_history, alpha)
    pipeline_results['algorithm'].append('RL-HEP')
    pipeline_results['power'].append(rl_power)
    pipeline_results['fdr'].append(rl_fdr)
    

    pipeline_results = pd.DataFrame(pipeline_results)
    pipeline_results.to_csv(f'../results/{dataset_name}/rl_vs_greedy_pipeline_results.csv', index=False)

    stepwise_results = pd.DataFrame(stepwise_results)
    stepwise_results.to_csv(f'../results/{dataset_name}/rl_vs_greedy_stepwise_results.csv', index=False)
