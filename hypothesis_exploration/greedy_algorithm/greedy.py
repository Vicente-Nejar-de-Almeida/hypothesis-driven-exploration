from queue import PriorityQueue
from hypothesis_exploration.user_data_model.data_model import Dataset, Group
from hypothesis_exploration.hypothesis_testing.hypothesis_test import HypothesisTest
from hypothesis_exploration.alpha_investing.covdiv_alpha import covdiv_alpha


class ExplorationNode:
    def __init__(self, g, h):
        self.g = g
        self.h = h
        self.children = []


class GreedyExplorer:
    def __init__(self, D: Dataset, H: [HypothesisTest], alpha: float, n: float, eta: float, lambd: float, w1: float, w2: float):
        self.D = D
        self.H = H
        self.alpha = alpha
        self.n = n
        self.eta = eta
        self.lambd = lambd
        self.w1 = w1
        self.w2 = w2

        self.reset()
    
    def reset(self):
        self.wealth = self.eta * self.alpha
        self.q = PriorityQueue()
        self.G_out_dict = {}
        self.request_history = []  # (str(g), h)
        self.run_covid_at_root()
    
    def run_covdiv(self, g_in, h):
        G_out, self.wealth, obj_value, cov_value, div_value, tested_requests = covdiv_alpha(D=self.D, g_in=g_in, h=h, alpha=self.alpha, n=self.n, wealth=self.wealth, lambd=self.lambd, w1=self.w1, w2=self.w2, request_history=self.request_history)
        self.q.put((-obj_value, (g_in, h)))
        self.G_out_dict[(g_in, h)] = G_out
        self.request_history += tested_requests
    
    def run_covid_at_root(self):
        g_in = Group(dataset=self.D, predicates={})
        for h in self.H:
            if self.wealth <= 0: return
            self.run_covdiv(g_in, h)
    
    def step(self):
        if not self.q.empty():
            _, (selected_g_in, previous_h) = self.q.get()
            selected_G_out = self.G_out_dict[(selected_g_in, previous_h)]
            for g_in in selected_G_out:
                for h in self.H:
                    if self.wealth <= 0: return
                    self.run_covdiv(g_in, h)
        else:
            selected_g_in = None
            selected_G_out = []
        return selected_g_in, selected_G_out
