import copy

import numpy as np


class GeneticAlgorithm:
    def __init__(self,
                 population_size,
                 crossover,
                 mutation,
                 elitism,
                 selection,
                 representation,
                 termination=None,
                 ):
        self.fit_history = []
        self.pop_history = []
        self.iteration = 0
        self.population_size = population_size
        self.problem = None
        self.crossover = crossover
        self.mutation = mutation
        self.elitism = elitism
        self.selection = selection
        self.representation = representation
        self.termination = termination
        self.population = []

    def run(self, problem, termination=None):
        self.termination = termination if termination is not None else self.termination
        self.problem = problem
        if self.termination is None:
            raise ValueError("Termination condition not specified")
        self.initialize()

        while self.has_next():
            self.next()

        return self.results()

    def initialize(self):
        self.population = self.representation.generate_population(self.population_size)
        self.iteration = 0

    def evaluate_population(self, population):
        decoded = self.representation.decode(population)
        fitness = self.problem.evaluate(decoded)
        return fitness

    def has_next(self):
        if self.termination.name == "maxiter" and self.iteration >= self.termination.max_iter:
            return False

        # TODO: implement other termination conditions

        return True

    def next(self):
        self.iteration += 1
        pop = self.population
        fitness = self.evaluate_population(pop)

        self.log(pop, fitness)

        if self.elitism is not None:
            self.elitism.select(pop, fitness)

        parents = self.selection.select_parents(pop, fitness)

        next_population = []
        parent_pairs = [[parents[i, :], parents[i + 1, :]] for i in range(0, len(parents), 2)]

        for parent_pair in parent_pairs:
            children = self.crossover.crossover(parent_pair)
            children = self.mutation.mutate(children)
            [next_population.append(c) for c in children]

        if self.elitism is not None:
            next_population = self.elitism.reinsert(next_population)

        self.population = np.array(next_population)

    def results(self):
        return {
            'population': self.representation.decode(self.population),
            'fitness': self.evaluate_population(self.population),
            'iteration': self.iteration,
            'pop_history': self.pop_history,
            'fit_history': self.fit_history,
        }

    def log(self, population, fitness):
        self.pop_history.append(copy.deepcopy(self.representation.decode(population)))
        self.fit_history.append(copy.deepcopy(fitness))
