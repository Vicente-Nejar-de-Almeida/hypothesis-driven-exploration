import numpy as np
from math import sqrt
from hypothesis_exploration.user_data_model import Dataset, Group, generate_candidates, coverage, diversity
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest


def cover_alpha(D: Dataset, g_in: Group, h: HypothesisTest, alpha: float, n: float, wealth: float, lambd: float) -> tuple[list, float, float, float, float]:
    candidate_groups = generate_candidates(g_in=g_in, dataset=D, min_sample_size=4)
    available_wealth = wealth
    alpha_star = available_wealth / (lambd + available_wealth)
    G_out = set()
    tested_requests = []
    
    while (len(candidate_groups) > 0) and (available_wealth > 0) and (len(G_out) < n):
        g_star_index = np.argmax([coverage(G_out.union({g}), g_in) for g in candidate_groups])
        g_star = candidate_groups.pop(g_star_index)
        current_alpha = alpha_star * sqrt(coverage({g_star}, g_in))

        if available_wealth - (current_alpha / (1 - current_alpha)) >= 0:
            if h.test(g_star.sample) <= current_alpha:
                available_wealth += alpha
                G_out.add(g_star)
            else:
                available_wealth -= (current_alpha / (1 - current_alpha))
            tested_requests.append((str(g_star), h))
    
    cov_value = coverage(G_out, g_in)
    div_value = diversity(G_out)
    
    return G_out, available_wealth, cov_value, div_value, tested_requests
