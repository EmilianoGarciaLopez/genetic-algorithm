from random import random

import numpy as np


class Mutation:
    def __init__(self, prob):
        self.prob = prob

    def mutate(self, individuals):
        return np.vstack(
            [self._mutate(individual) if random() < self.prob else individual for individual in individuals])

    def _mutate(self, individual):
        raise NotImplementedError


class BinaryMutation(Mutation):
    def mutate(self, individuals):
        return np.vstack([self._mutate(individual) for individual in individuals])

    def _mutate(self, individual):
        random_array = np.random.random(individual.shape[0])
        mask = (random_array < self.prob).astype(int)
        return np.abs(individual - mask)

# from src.core.factory import get_representation, get_problem
#
# if __name__ == "__main__":
#     problem = get_problem("rastrigin")
#     binary_rep = get_representation("binary", problem=problem, n_bits=10)
#     mutation = BinaryMutation(prob=0.1)
#     pop = binary_rep.generate_population(50)
#     mutated_pop = mutation.mutate(pop)
#
