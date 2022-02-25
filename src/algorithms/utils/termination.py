class Termination:
    def __init__(self, name):
        self.name = name


class MaxIter(Termination):
    def __init__(self, max_iter):
        super().__init__("maxiter")
        self.max_iter = max_iter

