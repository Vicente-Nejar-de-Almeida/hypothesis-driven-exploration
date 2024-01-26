import numpy as np
from math import sqrt
from hypothesis_exploration.user_data_model import Dataset, Group, generate_candidates, coverage, diversity
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest


def covdiv_alpha(D: Dataset, g_in: Group, h: HypothesisTest, alpha: float, n: float, wealth: float, lambd: float, w1: float, w2: float, request_history: list) -> tuple[list, float, float, float, float]:
    candidate_groups = [g for g in generate_candidates(g_in=g_in, dataset=D, min_sample_size=4) if (str(g), h) not in request_history]
    available_wealth = wealth
    alpha_star = available_wealth / (lambd + available_wealth)
    G_out = set()
    tested_requests = []
    previous_objective_value = 0
    while (len(candidate_groups) > 0) and (available_wealth > 0) and (len(G_out) < n):
        if len(G_out) == 0:
            candidate_obj_values = [coverage({g}, g_in) for g in candidate_groups]
            g_star_index = candidate_obj_values.index(np.percentile(candidate_obj_values, w1 * 100, method='nearest'))
        else:
            g_star_index = np.argmax([w1 * coverage(G_out.union({g}), g_in) + w2 * diversity(G_out.union({g})) for g in candidate_groups])
        g_star = candidate_groups.pop(g_star_index)
        
        new_obj_value = w1 * coverage(G_out.union({g_star}), g_in) + w2 * diversity(G_out.union({g_star}))
        obj_gain = new_obj_value - previous_objective_value
        previous_objective_value = new_obj_value

        current_alpha = alpha_star * sqrt(max(obj_gain, 0))
        if available_wealth - (current_alpha / (1 - current_alpha)) >= 0:
            if h.test(g_star.sample) <= current_alpha:
                available_wealth += alpha
                G_out.add(g_star)
            else:
                available_wealth -= (current_alpha / (1 - current_alpha))
            tested_requests.append((str(g_star), h))
    
    cov_value = coverage(G_out, g_in)
    div_value = diversity(G_out)
    obj_value = w1 * cov_value + w2 * div_value
    
    return G_out, available_wealth, obj_value, cov_value, div_value, tested_requests

"""
def covdiv_alpha(D: Dataset, g_in: Group, h: HypothesisTest, alpha: float, n: float, wealth: float, lambd: float, w1: float, w2: float, request_history: list) -> tuple[list, float, float, float, float]:
    candidate_groups = [g for g in generate_candidates(g_in=g_in, dataset=D, min_sample_size=4) if (str(g), h) not in request_history]
    available_wealth = wealth
    alpha_star = available_wealth / (lambd + available_wealth)
    G_out = set()
    obj_value = 0  # value of objective function
    cov_value = 0  # value of coverage of G_out
    div_value = 0  # value of diversity of G_out
    tested_requests = []
    while (len(candidate_groups) > 0) and (available_wealth > 0) and (len(G_out) < n):
        best_new_obj_value = 0
        cov_value_of_best = 0
        div_value_of_best = 0
        g_star = None
        for g in candidate_groups:
            cov = coverage(G_out.union({g}), g_in)
            div = diversity(G_out.union({g}))
            new_obj_value = w1 * cov + w2 * div
            if new_obj_value >= best_new_obj_value:
                g_star = g
                best_new_obj_value = new_obj_value
                cov_value_of_best = cov
                div_value_of_best = div
        candidate_groups.remove(g_star)
        obj_gain = best_new_obj_value - obj_value
        current_alpha = alpha_star * sqrt(max(obj_gain, 0))
        if available_wealth - (current_alpha / (1 - current_alpha)) >= 0:
            if h.test(g_star.sample) <= current_alpha:
                available_wealth += alpha
                G_out.add(g_star)
                obj_value = best_new_obj_value
                cov_value = cov_value_of_best
                div_value = div_value_of_best
            else:
                available_wealth -= (current_alpha / (1 - current_alpha))
            tested_requests.append((str(g_star), h))
    return G_out, available_wealth, obj_value, cov_value, div_value, tested_requests
"""
