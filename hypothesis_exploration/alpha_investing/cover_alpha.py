import numpy as np
from math import sqrt
from hypothesis_exploration.user_data_model import Dataset, Group, generate_candidates, coverage
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest


def cover_alpha(D: Dataset, g_in: Group, h: HypothesisTest, alpha: float, n: float, wealth: float, gamma: float, request_history: dict) -> tuple[list, float]:
    candidate_groups = generate_candidates(g_in=g_in, dataset=D)
    available_wealth = wealth
    alpha_star = available_wealth / (gamma + available_wealth)
    G_out = set()
    
    while (len(candidate_groups) > 0) and (available_wealth > 0) and (len(G_out) < n):
        g_star_index = np.argmax([coverage(G_out.union({g}), g_in) for g in candidate_groups])
        g_star = candidate_groups.pop(g_star_index)

        current_alpha = alpha_star * sqrt(coverage({g_star}, g_in))

        if (str(g_star), h) in request_history:
            if request_history[(str(g_star), h)][1]:
                G_out.add(g_star)
            continue

        if available_wealth - (current_alpha / (1 - current_alpha)) >= 0:
            pval = h.test(g_star.sample)
            if pval <= current_alpha:
                available_wealth += alpha
                G_out.add(g_star)
                request_history[(str(g_star), h)] = (pval, True)
            else:
                available_wealth -= (current_alpha / (1 - current_alpha))
                request_history[(str(g_star), h)] = (pval, False)
    
    return G_out, available_wealth
