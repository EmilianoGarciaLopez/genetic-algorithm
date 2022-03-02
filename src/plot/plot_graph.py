import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

from src.core.factory import get_problem, get_representation

pio.renderers.default = "browser"


def plot_function_animated(problem, pop_hist, sort=False, delta=0.5):
    print("Plotting function...")

    Z_matrix, x1, x2 = mesh_grind(delta, problem)

    contour_data = go.Contour(x=x1, y=x2, z=Z_matrix, colorscale='Viridis', opacity=0.5, visible=True)

    polar_pop_hist = np.array([convert_to_polar(pop) for pop in pop_hist])
    sort_population_history(pop_hist, polar_pop_hist if sort else None)

    frames = []
    for i, pop in enumerate(pop_hist):
        frames.append(go.Frame(data=[contour_data, go.Scatter(x=pop[:, 0],
                                                              y=pop[:, 1],
                                                              mode='markers',
                                                              opacity=1,
                                                              marker=dict(size=8),
                                                              ),
                                     ], layout=go.Layout(title_text=f'Generation {i}')))

    fig = go.Figure(
        data=[contour_data, contour_data],
        layout=go.Layout(
            xaxis=dict(range=[problem.lb, problem.ub], autorange=False),
            yaxis=dict(range=[problem.lb, problem.ub], autorange=False),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"transition": {"duration": 0}, "frame": {"redraw": False}}])])]
        ),
        frames=frames
    )

    fig.show()


def mesh_grind(delta, problem):
    x1 = np.arange(problem.lb, problem.ub, delta)
    x2 = np.arange(problem.lb, problem.ub, delta)
    X, Y = np.meshgrid(x1, x2)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    XY = np.vstack((X_flat, Y_flat)).T
    Z = problem.evaluate(XY)
    Z_matrix = Z.reshape(X.shape)
    return Z_matrix, x1, x2


def plot_function(problem, population=None, delta=0.1):
    print("Plotting function...")
    Z_matrix, x1, x2 = mesh_grind(delta, problem)

    fig = go.Figure(data=[go.Surface(x=x1, y=x2, z=Z_matrix, colorscale='Viridis', opacity=0.5)])

    if population is not None:
        fig.add_trace(
            go.Scatter3d(x=population[:, 0], y=population[:, 1], z=problem.evaluate(population), mode='markers',
                         marker=dict(size=6, color='red')))

    fig.show()


def sort_population_history(pop_hist, key_array=None):
    for i, pop in enumerate(pop_hist):
        for axis in [0, 1]:
            ixs = np.argsort(pop[:, axis] if key_array is None else key_array[i][:, axis], kind="stable")
            pop = pop[ixs]
            if key_array is not None:
                key_array[i] = key_array[i][ixs]
        pop_hist[i] = pop


def convert_to_polar(population):
    r = np.sqrt(population[:, 0] ** 2 + population[:, 1] ** 2)
    theta = np.arctan2(population[:, 1], population[:, 0])
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    return np.vstack((theta, r)).T

# if __name__ == '__main__':
#     problem = get_problem("rastrigin")
#     binary_rep = get_representation("binary", problem=problem, n_bits=16)
#     populations = np.array([binary_rep.decode(binary_rep.generate_population(100)) for i in range(20)])
#     polar_populations = np.array([convert_to_polar(pop) for pop in populations])
#     sort_population_history(populations, polar_populations)
