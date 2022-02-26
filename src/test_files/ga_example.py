from src.algorithms.genetic import GeneticAlgorithm
from src.algorithms.utils.termination import MaxIter
from src.core.factory import get_representation, get_problem, get_crossover, get_mutation, get_selection

if __name__ == "__main__":
    pop_size = 100
    problem = get_problem("rastrigin")
    binary_rep = get_representation("binary", problem=problem, n_bits=10)
    crossover = get_crossover("onepoint", prob=0.8)
    mutation = get_mutation("binary", prob=(1 / (2 * binary_rep.n_bits)))
    elitism = None
    selection = get_selection("roulette", n_selected=pop_size)
    termination = MaxIter(100)

    ga = GeneticAlgorithm(population_size=pop_size,
                          crossover=crossover,
                          mutation=mutation,
                          elitism=elitism,
                          selection=selection,
                          representation=binary_rep,
                          termination=termination, )

    results = ga.run(problem)

    print(min(results['fitness']))
