import numpy as np
import pandas as pd
from itertools import combinations


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, attributes: dict, multi_value_attribute_names: [str], action_dimension: str):
        self.dataframe = dataframe
        self.attributes = attributes  # {'att1': ['val1', 'val2', 'val3']}
        self.multi_value_attribute_names = multi_value_attribute_names
        self.action_dimension = action_dimension

        self.user_ids = set(dataframe.user_id.unique())

        self.action_dimension_min = dataframe[self.action_dimension].min()
        self.action_dimension_max = dataframe[self.action_dimension].max()

        USER_COUNT_LEN = 1
        ACTION_DESCRIPTION_LEN = 7
        ATT_VAL_LEN = sum([len(self.attributes[att]) for att in self.attributes])

        self.group_encoded_len = USER_COUNT_LEN + ACTION_DESCRIPTION_LEN + ATT_VAL_LEN
    
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
            action_dimension=self.action_dimension
        )


class Group:
    def __init__(self, dataset: Dataset, predicates: dict):
        self.predicates = predicates  # {'att1': 'val1', 'att2': 'val2'}
        self.parent_user_ids = dataset.user_ids
        self.dataset = dataset.filter(predicates=self.predicates)
        self.sample = self.dataset.dataframe[self.dataset.action_dimension].values
        self.user_ids = self.dataset.user_ids
    
    def __str__(self):
        if len(self.user_ids) == 0:
            return 'Empty group'
        
        str_of_predicates = []
        for att, val in sorted(self.predicates.items(), key=lambda p: p[0]):
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
        att_val_counts = []
        for att in self.dataset.attributes:
            for val in self.dataset.attributes[att]:
                att_val_counts.append(self.dataset.filter_dataframe({att: val})['user_id'].nunique() / len(self.dataset.user_ids))
        action_description = self.dataset.dataframe[self.dataset.action_dimension].describe()[1:]
        action_description = (action_description - self.dataset.action_dimension_min) / (self.dataset.action_dimension_max - self.dataset.action_dimension_min)
        
        self.encoded = np.array([user_count] + att_val_counts + action_description.tolist())
        return self.encoded


def generate_candidates(g_in: Group, dataset: Dataset, min_sample_size: dict = 1) -> [Group]:
    candidate_groups = []
    for att, possible_vals in dataset.attributes.items():
        if att not in g_in.predicates:
            for val in possible_vals:
                candidate_predicates = g_in.predicates.copy()  # shallow copy is enough, as keys and values are immutable
                candidate_predicates[att] = val
                candidate = Group(dataset=g_in.dataset, predicates=candidate_predicates)
                if len(candidate.sample) >= min_sample_size:
                    candidate_groups.append(candidate)
    return candidate_groups


# User group quality metrics

def coverage(G: [Group], g_in: Group) -> float:
    if len(g_in.user_ids) == 0:
        return 0
    return len(set([id for g in G for id in g.user_ids])) / len(g_in.user_ids)


def diversity(G: [Group]) -> float:
    if len(G) == 0:
        return 0
    else:
        penalty = 0
        for g1, g2 in combinations(G, 2):
            penalty += len(g1.user_ids.intersection(g2.user_ids))
        return 1 / (1 + penalty)
