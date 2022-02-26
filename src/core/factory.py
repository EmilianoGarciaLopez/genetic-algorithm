import re

from src.representations.encodings import BinaryRepresentation


def get_from_list(l, name, args, kwargs):
    i = None

    for k, e in enumerate(l):
        if e[0] == name:
            i = k
            break

    if i is None:
        for k, e in enumerate(l):
            if re.match(e[0], name):
                i = k
                break

    if i is not None:

        if len(l[i]) == 2:
            name, clazz = l[i]

        elif len(l[i]) == 3:
            name, clazz, default_kwargs = l[i]

            # overwrite the default if provided
            for key, val in kwargs.items():
                default_kwargs[key] = val
            kwargs = default_kwargs

        return clazz(*args, **kwargs)
    else:
        raise Exception("Object '%s' for not found in %s" % (name, [e[0] for e in l]))


def get_problem_options():
    from src.problems.test_functions import Rastrigin, Himmelblau
    INSTANCES = [
        ("rastrigin", Rastrigin),
        ("himmelblau", Himmelblau),
    ]

    return INSTANCES


def get_problem(name, *args, d={}, **kwargs):
    return get_from_list(get_problem_options(), name, args, {**d, **kwargs})


def get_selection_options():
    from src.algorithms.utils.selections import RouletteSelection
    INSTANCES = [
        ("roulette", RouletteSelection),
    ]

    return INSTANCES


def get_selection(name, *args, d={}, **kwargs):
    return get_from_list(get_selection_options(), name, args, {**d, **kwargs})


def get_crossover_options():
    from src.algorithms.utils.crossover import OnePointCrossover
    INSTANCES = [
        ("onepoint", OnePointCrossover),
    ]

    return INSTANCES


def get_crossover(name, *args, d={}, **kwargs):
    return get_from_list(get_crossover_options(), name, args, {**d, **kwargs})


def get_mutation_options():
    from src.algorithms.utils.mutation import BinaryMutation
    INSTANCES = [
        ("binary", BinaryMutation),
    ]

    return INSTANCES


def get_mutation(name, *args, d={}, **kwargs):
    return get_from_list(get_mutation_options(), name, args, {**d, **kwargs})


def get_representation_options():
    from src.problems.test_functions import Rastrigin, Himmelblau
    instances = [
        ("binary", BinaryRepresentation),
        # ("decimal", DecimalRepresentation),
    ]

    return instances


def get_representation(name, *args, d={}, **kwargs):
    return get_from_list(get_representation_options(), name, args, {**d, **kwargs})
