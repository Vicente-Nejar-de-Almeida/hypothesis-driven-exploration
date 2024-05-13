import numpy as np
import pandas as pd
from itertools import combinations

from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, attributes: dict, multi_value_attribute_names: list[str], action_dimension: str, action_dimension_min: float, action_dimension_max: float):
        self.dataframe = dataframe
        self.attributes = attributes  # {'att1': ['val1', 'val2', 'val3']}
        self.multi_value_attribute_names = multi_value_attribute_names
        self.action_dimension = action_dimension

        self.user_ids = set(dataframe.user_id.unique())

        self.action_dimension_min = action_dimension_min
        self.action_dimension_max = action_dimension_max

        USER_COUNT_LEN = 1
        # ATT_BOOL_VAL = sum([len(self.attributes[att]) for att in self.attributes])
        ATT_BOOL_VAL = len(self.attributes)
        ACTION_DESCRIPTION_LEN = 7

        self.group_encoded_len = USER_COUNT_LEN + ATT_BOOL_VAL + ACTION_DESCRIPTION_LEN
    
    def filter_dataframe(self, predicates: set) ->  pd.DataFrame:
        if len(predicates) > 0:
            queries = [f'{att} == "{val}"' for att, val in predicates.items() if val in self.attributes[att] and att not in self.multi_value_attribute_names]
            queries += [f'{att}.str.contains("{val}", regex=False)' for att, val in predicates.items() if val in self.attributes[att] and att in self.multi_value_attribute_names]
            query = ' & '.join(queries)
            return self.dataframe.query(query, engine='python')
        else:
            return self.dataframe
    
    def filter(self, predicates: set):
        filtered_dataframe = self.filter_dataframe(predicates=predicates)
        return Dataset(
            dataframe=filtered_dataframe,
            attributes=self.attributes,
            multi_value_attribute_names=self.multi_value_attribute_names,
            action_dimension=self.action_dimension,
            action_dimension_min=self.action_dimension_min,
            action_dimension_max=self.action_dimension_max,
        )


class Group:
    def __init__(self, dataset: Dataset, predicates: dict):
        self.predicates = predicates  # {'att1': 'val1', 'att2': 'val2'}
        self.parent_user_ids = dataset.user_ids
        self.dataset = dataset.filter(predicates=self.predicates)
        self.sample = self.dataset.dataframe[self.dataset.action_dimension].values
        self.user_ids = self.dataset.user_ids
        self.encoded = self.encode()
    
    def __str__(self):
        if len(self.user_ids) == 0:
            return 'Empty group'
        
        str_of_predicates = []
        # for att, val in self.predicates.items():
        for att, val in sorted(self.predicates.items(), key=lambda p: p[0]):
            str_of_predicates.append(f'{att}:{val}')
        if len(str_of_predicates) > 0:
            return '|'.join(str_of_predicates)
        else:
            return 'All users'
    
    def to_string_in_exploration_order(self):
        if len(self.user_ids) == 0:
            return 'Empty group'
        
        str_of_predicates = []
        for att, val in self.predicates.items():
        # for att, val in sorted(self.predicates.items(), key=lambda p: p[0]):
            str_of_predicates.append(f'{att}:{val}')
        if len(str_of_predicates) > 0:
            return '|'.join(str_of_predicates)
        else:
            return 'All users'
    
    def __lt__(self, other):
        return len(self.sample) < len(other.sample)
    
    def encode(self):

        if len(self.user_ids) == 0:
            return np.zeros(self.dataset.group_encoded_len)
        
        user_count = len(self.user_ids) / len(self.parent_user_ids)
        # user_count = len(self.user_ids) / len(self.dataset.user_ids)
        att_bool_vals = []
        for att in self.dataset.attributes:
            if att in self.predicates:
                att_bool_vals.append(1)
            else:
                att_bool_vals.append(0)
            """
            for val in self.dataset.attributes[att]:
                if att in self.predicates and self.predicates[att] == val:
                    att_bool_vals.append(1)
                else:
                    att_bool_vals.append(0)
            """
        
        action_description = self.dataset.dataframe[self.dataset.action_dimension].describe()[1:]
        action_description = (action_description - self.dataset.action_dimension_min) / (self.dataset.action_dimension_max - self.dataset.action_dimension_min)
        
        encoded = np.array([user_count] + att_bool_vals + action_description.tolist())
        return encoded


def generate_candidates(g_in: Group, dataset: Dataset, min_sample_size: int = 20, min_user_count: int = 20) -> list[Group]:
    candidate_groups = []
    for att, possible_vals in dataset.attributes.items():
        if att not in g_in.predicates:
            for val in possible_vals:
                candidate_predicates = g_in.predicates.copy()  # shallow copy is enough, as keys and values are immutable
                candidate_predicates[att] = val
                candidate = Group(dataset=g_in.dataset, predicates=candidate_predicates)
                if (len(candidate.sample) >= min_sample_size) and (len(candidate.user_ids) >= min_user_count):
                    candidate_groups.append(candidate)
    return candidate_groups


# User group quality metrics

def coverage(G: list[Group], g_in: Group) -> float:
    if len(g_in.user_ids) == 0:
        return 0
    return len(set([id for g in G for id in g.user_ids])) / len(g_in.user_ids)


"""
def diversity(G: list[Group]) -> float:
    if len(G) == 0:
        return 0
    else:
        penalty = 0
        for g1, g2 in combinations(G, 2):
            penalty += len(g1.user_ids.intersection(g2.user_ids))
        return 1 / (1 + penalty)
"""


def jaccard_coefficient(A: set, B: set):
    return (len(A.intersection(B)) / len(A.union(B)))


def jaccard_distance(A: set, B: set):
    return 1 - jaccard_coefficient(A, B)


def diversity(G: list[Group], normalized: bool = False):
    if len(G) <= 1:
        return 0
    
    G = list(G)
    
    diversities = []
    for i in range(len(G) - 1):
        for j in range(i+1, len(G)):
            g1 = G[i]
            g2 = G[j]
            diversities.append(jaccard_distance(g1.user_ids, g2.user_ids))
    
    if normalized:
        return np.mean(diversities)
    else:
        return sum(diversities)
