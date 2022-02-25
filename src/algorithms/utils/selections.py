import numpy as np

from src.core.factory import get_problem, get_representation


class Selection:
    def __init__(self, n_selected):
        self.n_selected = n_selected

    def select_parents(self, population, fitness):
        return self._select(population, fitness)

    def _select(self, population, fitness):
        raise NotImplementedError


class RouletteSelection(Selection):

    def _select(self, population, fitness):
        if self.n_selected is None:
            self.n_selected = population.shape[0]
        weights = fitness / sum(fitness)
        ixs = list(range(population.shape[0]))
        selected = np.random.choice(ixs, self.n_selected, p=weights)
        return population[selected, :]


if __name__ == '__main__':
    problem = get_problem("rastrigin")
    binary_rep = get_representation("binary", problem=problem, n_bits=10)

    pop = binary_rep.generate_population(50)
    selection = RouletteSelection(n_selected=50)

    parents = selection.select_parents(pop, np.random.rand(50))
