import numpy as np


class Problem:
    def __init__(self, name, fx, lb, ub, dim, f_opt):
        self.name = name
        self.fx = fx
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.f_opt = f_opt
        self.function_evals = 0

    def evaluate(self, x):
        self.function_evals += x.shape[0]
        y = np.apply_along_axis(self.fx, 1, x)
        return y

    def get_name(self):
        return self.name


class Rastrigin(Problem):
    def __init__(self, dim=2):
        super().__init__(name='Rastrigin', fx=self.rastrigin, lb=-5.12, ub=5.12, dim=dim, f_opt=0)

    def rastrigin(self, X, A=10):
        return A * X.shape[0] + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=0)


class Himmelblau(Problem):
    def __init__(self, dim=2):
        super().__init__(name='Himmelblau', fx=self.himmelblau, lb=-6, ub=6, dim=dim, f_opt=0)

    def himmelblau(self, X):
        return (X[0] ** 2 + X[1] - 11) ** 2 + (X[0] + X[1] ** 2 - 7) ** 2
