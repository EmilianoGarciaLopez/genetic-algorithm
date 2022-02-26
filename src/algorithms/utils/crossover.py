from random import random
import numpy as np


class Crossover:
    def __init__(self, prob):
        self.prob = prob

    def crossover(self, parents):
        if self.prob > random():
            return self._crossover(parents)
        else:
            return parents

    def _crossover(self, parents):
        raise NotImplementedError


class OnePointCrossover(Crossover):
    def _crossover(self, parents):
        parent1, parent2 = parents[0], parents[1]
        point = int(random() * parent1.shape[0])
        child1 = np.hstack([parent1[:point], parent2[point:]])
        child2 = np.hstack([parent2[:point], parent1[point:]])

        return np.vstack([child1, child2])
