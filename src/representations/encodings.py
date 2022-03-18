import numpy as np

from src.problems.test_functions import Rastrigin


class Representation:
    def __init__(self, name, lb=None, ub=None):
        self.name = name
        self.lb = lb
        self.ub = ub

    def encode(self, x):
        # return [self._encode(xi) for xi in x]
        return np.vstack([self._encode(xi) for xi in x])

    def decode(self, x):
        return np.apply_along_axis(self._decode, 1, x)

    def _encode(self, x):
        return x

    def _decode(self, x):
        return x


class BinaryRepresentation(Representation):
    def __init__(self, problem, n_bits=8):
        super().__init__(name="binary", lb=problem.lb, ub=problem.ub)
        self.n_bits = n_bits
        self.binary_ub = 2 ** self.n_bits - 1
        self.n_vars = problem.dim

    def split_binary_array(self, binary_array):
        return np.split(np.array(binary_array), self.n_vars, axis=0)

    def _encode(self, x):
        if len(x) != self.n_vars:
            raise ValueError("x must have the same length as n_vars")
        bin_x = []
        for xi in x:
            percentage = (xi - self.lb) / (self.ub - self.lb)
            bin_x.append(np.array(list(format(int(percentage * self.binary_ub), f"0{self.n_bits}b"))).astype(int))
        return np.hstack(bin_x)

    def _decode(self, x):
        bin_vars = self.split_binary_array(x)
        dec_vars = []
        for bin_var in bin_vars:
            decimal = int("".join(map(str, bin_var)), 2)
            dec_vars.append(decimal / self.binary_ub * (self.ub - self.lb) + self.lb)
        return np.array(dec_vars)

    def generate_population(self, population_size):
        random_decimal_array = np.random.random(size=(population_size, self.n_vars)) * (self.ub - self.lb) + self.lb
        return self.encode(random_decimal_array)

# if __name__ == "__main__":
#     problem = Rastrigin()
#     bin_representation = BinaryRepresentation(problem)
#     pop = bin_representation.generate_population(10)
#     print(bin_representation.split_binary_array([0, 0, 0, 1, 0, 1, 0, 0]))
#     encoded = bin_representation.encode(np.array([[2, 2], [5, 5]]))
#     decoded = bin_representation.decode(encoded)
