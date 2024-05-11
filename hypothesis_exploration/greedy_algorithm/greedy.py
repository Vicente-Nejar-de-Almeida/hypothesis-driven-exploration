from queue import PriorityQueue
from hypothesis_exploration.user_data_model.data_model import Dataset, Group, coverage, diversity
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest
from hypothesis_exploration.alpha_investing import covdiv_alpha, cover_alpha


class ExplorationNode:
    def __init__(self, g, h):
        self.g = g
        self.h = h
        self.children = []


class GreedyExplorer:
    def __init__(self, D: Dataset, H: list[HypothesisTest], alpha: float, n: float, eta: float, gamma: float, lambd: float, starting_predicates: dict = {}):
        self.D = D
        self.H = H
        self.alpha = alpha
        self.n = n
        self.eta = eta
        self.gamma = gamma
        self.lambd = lambd
        self.starting_predicates = starting_predicates

        self.reset()
    
    def reset(self):
        self.wealth = self.eta * self.alpha
        self.q = PriorityQueue()
        self.G_out_dict = {}
        self.request_history = {}
        self.run_covdiv_at_root()
    
    def run_covdiv(self, g_in, h):
        G_out, self.wealth = covdiv_alpha(D=self.D, g_in=g_in, h=h, alpha=self.alpha, n=self.n, wealth=self.wealth, gamma=self.gamma, lambd=self.lambd, request_history=self.request_history)
        obj_value = coverage(G_out, g_in) + self.lambd * diversity(G_out)
        self.q.put((-obj_value, (g_in, h)))
        self.G_out_dict[(g_in, h)] = G_out
    
    def run_covdiv_at_root(self):
        g_in = Group(dataset=self.D, predicates=self.starting_predicates)
        for h in self.H:
            if self.wealth <= 0: return
            self.run_covdiv(g_in, h)
    
    def step(self):
        if not self.q.empty():
            _, (selected_g_in, previous_h) = self.q.get()
            selected_G_out = self.G_out_dict[(selected_g_in, previous_h)]
            for g_in in selected_G_out:
                for h in self.H:
                    if self.wealth <= 0:
                        return None, [], None
                    self.run_covdiv(g_in, h)
        else:
            selected_g_in = None
            selected_G_out = []
            previous_h = None
        return selected_g_in, selected_G_out, previous_h
