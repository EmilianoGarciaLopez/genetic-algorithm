from src.algorithms.genetic import GeneticAlgorithm
from src.algorithms.utils.termination import MaxIter
from src.core.factory import get_representation, get_problem

if __name__ == "__main__":
    problem = get_problem("rastrigin")
    binary_rep = get_representation("binary", problem=problem, n_bits=10)
    crossover = None
    mutation = None
    elitism = None
    selection = None
    termination = MaxIter(10)

    ga = GeneticAlgorithm(population_size=100,
                          crossover=crossover,
                          mutation=mutation,
                          elitism=elitism,
                          selection=selection,
                          representation=binary_rep,
                          termination=termination, )

    ga.run(problem)
    print(ga.population)
