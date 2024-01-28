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

group = Group(dataset=dataset, predicates={})
print('All users:', group.encode())

# females = Group(dataset=dataset, predicates={'gender': 'F'})
# print('Female users:', females.encode())

# print(len(group.encode()), dataset.group_encoded_len)

empty_dataframe = dataframe.head(0)
empty_dataset = Dataset(
    dataframe=empty_dataframe,
    attributes=dataset.attributes,
    multi_value_attribute_names=dataset.multi_value_attribute_names,
    action_dimension=dataset.action_dimension,
)

empty_group = Group(dataset=empty_dataset, predicates={})
print('Empty dataset:', empty_group.encode())
