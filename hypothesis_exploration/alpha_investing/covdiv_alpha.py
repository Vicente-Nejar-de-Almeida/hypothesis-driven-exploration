import numpy as np
from math import sqrt
from hypothesis_exploration.user_data_model import Dataset, Group, generate_candidates, coverage, jaccard_distance
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest


def proportion_of_new_users(new_group: Group, G: list[Group]):
    if len(new_group.user_ids) == 0:
        return 0
    
    G_user_ids = set([id for g in G for id in g.user_ids])
    new_user_count = 0
    for uid in new_group.user_ids:
        if uid not in G_user_ids:
            new_user_count += 1
    
    return new_user_count / len(new_group.user_ids)


def diversity_among_candidates(g, C):
    if len(C) == 0:
        return 0
    
    diversities = []
    for c in C:
        diversities.append(jaccard_distance(g.user_ids, c.user_ids))
    else:
        return np.mean(diversities)


def marginal_gain_on_diversity(g, G):
    if len(G) == 0:
        return 0
    
    return sum([jaccard_distance(g.user_ids, g2.user_ids) for g2 in G])


def marginal_gain_on_coverage(g, G, g_in):
    return coverage(G.union({g}), g_in) - coverage(G, g_in)


def covdiv_alpha(D: Dataset, g_in: Group, h: HypothesisTest, alpha: float, n: float, wealth: float, gamma: float, lambd: float, request_history: dict) -> tuple[list, float]:
    candidate_groups = generate_candidates(g_in=g_in, dataset=D)
    available_wealth = wealth
    alpha_star = available_wealth / (gamma + available_wealth)
    G_out = set()

    while (len(candidate_groups) > 0) and (available_wealth > 0) and (len(G_out) < n):
        g_star_index = np.argmax([(1/2) * marginal_gain_on_coverage(g, G_out, g_in) + lambd * marginal_gain_on_diversity(g, G_out) for g in candidate_groups])
        """
        if len(G_out) == 0:
            g_star_index = np.argmax([(1/2) * marginal_gain_on_coverage(g, G_out, g_in) + lambd * diversity_among_candidates(g, [g2 for g2 in candidate_groups if g2 != g]) for g in candidate_groups])
        else:
            g_star_index = np.argmax([(1/2) * marginal_gain_on_coverage(g, G_out, g_in) + lambd * marginal_gain_on_diversity(g, G_out) for g in candidate_groups])
        """
        
        g_star = candidate_groups.pop(g_star_index)

        x = (1 / (1 + lambd))
        current_alpha = alpha_star * sqrt(x * coverage({g_star}, g_in) + lambd * x * proportion_of_new_users(g_star, G_out))
        
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
