import numpy as np


class Elitism:
    def __init__(self, prob):
        self.n = None
        self.selected = None
        self.elitism_rate = prob

    def select(self, population, fitness):
        # sort the population by fitness
        sorted_ix = np.argsort(fitness)
        self.n = int(self.elitism_rate * len(population))
        self.selected = population[sorted_ix[:self.n], :]

    def reinsert(self, population):
        population = np.array(population)
        random_indexes = np.random.choice(list(range(population.shape[0])), size=self.n, replace=False)

        population[random_indexes, :] = self.selected
        return population

# if __name__ == "__main__":
#     import copy
#     from src.core.factory import get_representation, get_problem
#
#     problem = get_problem("rastrigin")
#     binary_rep = get_representation("binary", problem=problem, n_bits=10)
#     pop = binary_rep.generate_population(20)
#     pop0 = copy.deepcopy(pop)
#     fitness = problem.evaluate(pop)
#     elitism = Elitism(0.1)
#     elitism.select(pop, fitness)
#     new_pop = elitism.reinsert(pop)
#
#     print(pop0 - new_pop)
