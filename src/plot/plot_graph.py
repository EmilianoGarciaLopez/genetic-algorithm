from src.core.factory import get_problem
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

from src.core.factory import get_problem

pio.renderers.default = "browser"


def plot_function(problem, delta=0.05):
    print("Plotting function...")

    x1 = np.arange(problem.lb, problem.ub, delta)
    x2 = np.arange(problem.lb, problem.ub, delta)

    X, Y = np.meshgrid(x1, x2)

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    XY = np.vstack((X_flat, Y_flat)).T
    Z = problem.evaluate(XY)
    Z_matrix = Z.reshape(X.shape)

    fig = go.Figure(data=[go.Surface(x=x1, y=x2, z=Z_matrix, colorscale='Viridis', opacity=0.75)])
    fig.show()


if __name__ == "__main__":
    rastrigin = get_problem("rastrigin")
    himmelblau = get_problem("himmelblau", dim=2)
    # plot_function(rastrigin)
    plot_function(rastrigin)
    # print(himmelblau.function_evals)
    # plot_function(rastrigin)
