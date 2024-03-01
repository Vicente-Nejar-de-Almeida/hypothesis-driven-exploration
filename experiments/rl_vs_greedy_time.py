import time
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.rl import GroupExplorationEnv, TrueOnlineSarsaLambda
from hypothesis_exploration.greedy_algorithm import GreedyExplorer

from datasets.MovieLens import params as movie_params

movie_dataframe = pd.read_csv('../datasets/MovieLens/MovieLens.csv')

movie_dataset = Dataset(
    dataframe=movie_dataframe,
    multi_value_attribute_names=movie_params.multi_value_attribute_names,
    attributes=movie_params.attributes,
    action_dimension=movie_params.action_dimension,
    action_dimension_min=movie_params.action_dimension_min,
    action_dimension_max=movie_params.action_dimension_max
)

movie_hypotheses = [
    [
        HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    ],

    [
        HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
        HypothesisTest(aggregation='variance', null_value=1, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
    ],

    [
        HypothesisTest(aggregation='mean', null_value=3.5, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
        HypothesisTest(aggregation='variance', null_value=1, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE),
        HypothesisTest(aggregation='distribution', null_value='norm', alternative='two-sided', n_sample=HypothesisTest.ONE_SAMPLE),
    ],
]

rl_runs = 1

n_values = [3, 5, 7]
m_values = [5, 10, 15]

eta = 1
alpha = 0.05
gamma = 500
lambd = 1
initial_wealth = eta * alpha

time_results = {
    'algorithm': [],
    'H': [],
    'm': [],
    'n': [],
    'time': [],
}

for H in movie_hypotheses:
    print(f'H: {len(H)}')
    for m in m_values:
        print(f'm: {m}')
        for n in n_values:
            print(f'n: {n}')
            time_greedy = 0

            print('\tRunning Greedy-HEP')
            greedy = GreedyExplorer(
                D=movie_dataset,
                H=H,
                alpha=alpha,
                n=n,
                eta=eta,
                gamma=gamma,
                lambd=lambd,
            )

            t0_greedy = time.time()
            greedy.reset()
            t1_greedy = time.time()
            time_greedy += (t1_greedy - t0_greedy)

            for step in range(m):
                print(f'\t\tStep {step}')
                t0_greedy = time.time()
                selected_g_in, selected_G_out, selected_h = greedy.step()
                t1_greedy = time.time()
                time_greedy += (t1_greedy - t0_greedy)
                if selected_g_in is None:
                    print('\t\tBreaking')
                    break
            
            print(f'\t\t\tTime: {time_greedy} seconds')
            
            time_results['algorithm'].append('Greedy-HEP')
            time_results['H'].append(len(H))
            time_results['m'].append(m)
            time_results['n'].append(n)
            time_results['time'].append(time_greedy)
            
            print('\tRunning RL-HEP')
            env = GroupExplorationEnv(
                D=movie_dataset,
                H=H,
                alpha=alpha,
                n=n,
                m=m,
                eta=eta,
                gamma=gamma,
                lambd=lambd,
                additional_group_count=0,
                max_predicates=4,
            )

            obs, info = env.reset()
            agent = TrueOnlineSarsaLambda(
                env.observation_space,
                env.action_space,
                alpha=0.001,
                fourier_order=2,
                gamma=1,
                lamb=0.9,
                epsilon=0.0,
                min_max_norm=True
            )

            times_rl = []
            
            for run in range(rl_runs):
                print(f'\t\tRun {run}')
                single_run_time_rl = 0
                obs, info = env.reset()
                done = False
                step = 0
                while not done:
                    print(f'\t\t\tStep {step}')
                    t0_rl = time.time()
                    action = agent.act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = (terminated or truncated)
                    t1_rl = time.time()
                    single_run_time_rl += (t1_rl - t0_rl)
                    step += 1
                print(f'\t\t\tTime: {single_run_time_rl} seconds')
                times_rl.append(single_run_time_rl)

            time_rl = np.mean(times_rl)

            time_results['algorithm'].append('RL-HEP')
            time_results['H'].append(len(H))
            time_results['m'].append(m)
            time_results['n'].append(n)
            time_results['time'].append(time_rl)


time_results = pd.DataFrame(time_results)
time_results.to_csv(f'../results/MovieLens/rl_vs_greedy_time_results.csv', index=False)
