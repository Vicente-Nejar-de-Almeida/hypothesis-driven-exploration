import pandas as pd

from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append("..")

from hypothesis_exploration.user_data_model import Dataset, coverage, diversity
from hypothesis_exploration.hypothesis_testing import HypothesisTest
from hypothesis_exploration.rl import GroupExplorationEnv
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

hypothesis = HypothesisTest(aggregation='mean', null_value=3, alternative='greater', n_sample=HypothesisTest.ONE_SAMPLE)

eta = 1
alpha = 0.05
lambd = 500
# n_values = [2, 4, 6, 8, 10]
n_values = [4]
initial_wealth = eta * alpha
# weight_values = [(1, 0), (0, 1), (0.25, 0.75), (0.75, 0.25), (0.5, 0.5)]
weight_values = [(0.5, 0.5)]

results = {
    'step': [],
    'n': [],
    'w1': [],
    'w2': [],
    'algorithm': [],
    'g_in': [],
    'G_out': [],
    'coverage': [],
    'diversity': [],
    'objective': [],
    # 'time': []
}

for n in n_values:
    print('n:', n)

    for weights in weight_values:
        w1, w2 = weights
        print('w:', weights)

        def make_env():
            return GroupExplorationEnv(
                D=dataset,
                H=[hypothesis],
                alpha=alpha,
                n=n,
                eta=eta,
                lambd=lambd,
                w1=w1,
                w2=w2,
            )


        env = DummyVecEnv([make_env])
        model = MaskablePPO("MultiInputPolicy", env, gamma=1, seed=32, verbose=1, n_steps=10)
        model.learn(total_timesteps=50, log_interval=1)

        env = make_env()
        obs, info = env.reset()
        g_in_rl = []
        G_out_rl = []

        done = False
        step = 0
        while not done:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            print(info)
            g_in_rl.append(info['g_in'])
            G_out_rl.append(info['G_out'])
            results['step'].append(step)
            results['n'].append(n)
            results['w1'].append(w1)
            results['w2'].append(w2)
            results['algorithm'].append('rl')
            results['g_in'].append(info['g_in_str'])
            results['G_out'].append('; '.join(info['G_out_str']))
            cov = coverage(info['G_out'], info['g_in'])
            div = diversity(info['G_out'])
            obj = w1 * cov + w2 * div
            results['coverage'].append(cov)
            results['diversity'].append(div)
            results['objective'].append(obj)
            step += 1
            done = terminated or truncated
        
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

        g_in_greedy = []
        G_out_greedy = []

        steps = len(g_in_rl)
        for step in range(steps):
            selected_g_in, selected_G_out = greedy.step()
            g_in_greedy.append(selected_g_in)
            G_out_greedy.append(selected_G_out)
            results['step'].append(step)
            results['n'].append(n)
            results['w1'].append(w1)
            results['w2'].append(w2)
            results['algorithm'].append('greedy')
            results['g_in'].append(str(selected_g_in))
            results['G_out'].append('; '.join([str(g) for g in selected_G_out]))
            cov = coverage(selected_G_out, selected_g_in)
            div = diversity(selected_G_out)
            obj = w1 * cov + w2 * div
            results['coverage'].append(cov)
            results['diversity'].append(div)
            results['objective'].append(obj)
            print(f'Step {step}, wealth: {greedy.wealth}')
            print(selected_g_in)
            for g in selected_G_out:
                print('\t' + str(g))

results = pd.DataFrame(results)
results.to_csv('../results/MovieLens/rl_vs_greedy.csv', index=False)
