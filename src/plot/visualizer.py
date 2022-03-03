from plotly.subplots import make_subplots

import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

# import plotly.express as px
# colors = px.colors.qualitative.Plotly

pio.renderers.default = "browser"


class Visualizer:
    """
    Taken from https://github.com/ghis9917/UM_ARS_G8/blob/master/01_PSO/src/Visualizer.py
    """

    def __init__(self,
                 problem,
                 pop_hist,
                 fit_hist,
                 title=None):

        # Read data from a csv
        X, Y, Z = self.create_map_variables(delta=0.05, problem=problem)

        layout = self.create_layout(title if title is not None else "GA Visualization")

        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "surface"}, {"type": "scatter"}]])

        fig = fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=False, opacity=0.25),
            row=1, col=1
        )

        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
                          row=1, col=1)

        fig = fig.add_trace(
            go.Scatter3d(
                x=pop_hist[0][:, 0],
                y=pop_hist[0][:, 1],
                z=fit_hist[0],
                # hovertext=[f'team {data.get(0)[i].get("swarm")}' for i in range(len(data.get(0)))],
                # hoverinfo="text",
                mode="markers",
                name=f'Population',
                marker=dict(
                    # color=[Color(colors[particle_info.get("swarm")], luminance=0.3).get_hex() if particle_info.get(
                    #     "best") else colors[particle_info.get("swarm")] for particle_info in data.get(0)],
                    size=10)
            ),
            row=1, col=1
        )

        fig = fig.add_trace(
            go.Scatter3d(
                x=pop_hist[0][:, 0],
                y=pop_hist[0][:, 1],
                z=[-2.5 for _ in range(len(fit_hist[0]))],  # TODO: find value
                # hovertext=[f'team {data.get(0)[i].get("swarm")}' for i in range(len(data.get(0)))],
                # hoverinfo="text",
                mode="markers",
                showlegend=False,
                marker=dict(
                    # color=[
                    #     Color(colors[particle_info.get("swarm")], luminance=0.3).get_hex()
                    #     if particle_info.get("best") else colors[particle_info.get("swarm")]
                    #     for particle_info in data.get(0)],
                    size=5)
            ),
            row=1, col=1
        )

        # Add each line chart to the figure
        # counter = 0
        for key, np_fun in zip(["avg", "best", "worst"], [np.mean, np.max, np.min]):
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[np_fun(fit_hist[0])],
                    mode='lines',
                    name=key
                ),
                row=1, col=2
            )
            # counter += 1

        sliders_dict = self.create_slider(pop_hist)

        frames = [
            go.Frame(
                data=self.get_current_data_frame(k, pop_hist, fit_hist),
                name=str(k),
                traces=list(range(1, len(fig.data)))
            )
            for k in range(pop_hist.shape[0])
        ]

        fig.frames = frames
        layout["sliders"] = [sliders_dict]

        fig.update_layout(layout)
        fig.update_layout(coloraxis_showscale=False)
        # fig.update_xaxes(range=[problem.lb, problem.ub], row=1, col=1)
        # fig.update_yaxes(range=[problem.lb, problem.ub], row=1, col=1)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[problem.lb, problem.ub], ),
                yaxis=dict(range=[problem.lb, problem.ub], ),
                zaxis=dict(range=[min(-3, np.min(np.min(Z))), np.max(np.max(Z))], ), ))
        # fig.update_zaxes(range=[np.min(np.min(Z)), np.max(np.max(Z))], row=1, col=1)

        fig.update_xaxes(range=[0, pop_hist.shape[0]], row=1, col=2)
        fig.update_yaxes(range=[0, np.max(np.max(fit_hist)) + 5], row=1, col=2)
        self.fig = fig
        # fig.show()
        # fig.write_html("index.html", include_plotlyjs='cdn', include_mathjax=False, auto_play=False)

    def create_slider(self, pop_hist):
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Generation:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": dict(duration=5, easing='linear'),
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [
                    [k],
                    {"frame": dict(duration=500, redraw=True),
                     "mode": "immediate",
                     "transition": dict(duration=0, easing='linear')}
                ],
                    "label": k,
                    "method": "animate"} for k in range(pop_hist.shape[0])
            ]
        }
        return sliders_dict

    def show_fig(self):
        self.fig.show()

    def write_fig(self, write_title: str):
        self.fig.write_html(f'{write_title}.html', include_plotlyjs='cdn', include_mathjax=False, auto_play=False)

    def get_current_data_frame(self, gen, pop_hist, fit_hist):
        ret_list = [
            go.Scatter3d(
                x=pop_hist[gen][:, 0],
                y=pop_hist[gen][:, 1],
                z=fit_hist[gen],
                # hovertext=[f'team {pop_hist.get(gen)[i].get("swarm")}' for i in range(len(pop_hist.get(0)))],
                # hoverinfo="text",
                mode="markers",
                marker=dict(
                    # color=[Color(colors[particle_info.get("swarm")], luminance=0.3).get_hex() if particle_info.get(
                    #     "best") else colors[particle_info.get("swarm")] for particle_info in pop_hist.get(gen)],
                    size=10)),
            go.Scatter3d(
                x=pop_hist[gen][:, 0],
                y=pop_hist[gen][:, 1],
                z=[-2.5 for _ in range(len(fit_hist[gen]))],  # TODO: find value
                # hovertext=[f'team {pop_hist.get(gen)[i].get("swarm")}' for i in range(len(pop_hist.get(0)))],
                # hoverinfo="text",
                mode="markers",
                marker=dict(
                    # color=[Color(colors[particle_info.get("swarm")], luminance=0.3).get_hex() if particle_info.get(
                    #     "best") else colors[particle_info.get("swarm")] for particle_info in pop_hist.get(gen)],
                    size=5))
        ]

        for key, np_fun in zip(["avg", "best", "worst"], [np.mean, np.max, np.min]):
            ret_list.append(
                go.Scatter(
                    x=list(range(gen)),
                    y=np_fun(fit_hist[:gen, :], axis=1),
                    mode='lines',
                    name=key
                )
            )
        return ret_list

    def create_layout(self, title):
        layout = go.Layout(
            title_text=title,
            updatemenus=[dict(type="buttons",
                              buttons=[
                                  dict(label="Play",
                                       method="animate",
                                       args=[None, dict(frame=dict(duration=200, redraw=True),
                                                        fromcurrent=True,
                                                        transition=dict(duration=0, easing='linear')
                                                        )]
                                       ),
                                  dict(label="Pause",
                                       method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=0, redraw=False),
                                                  mode='immediate',
                                                  transition=dict(duration=0)
                                                  )],
                                       )
                              ],
                              direction="left",
                              pad={"r": 10, "t": 87},
                              showactive=False,
                              x=0.1,
                              xanchor="right",
                              y=0,
                              yanchor="top"
                              )]
        )
        return layout

    # @staticmethod
    # def create_map_variables(opti_func):
    #     x_y_range = np.linspace(Const.MIN_POS, Const.MAX_POS, Const.grid_granularity)
    #     x_y_range = np.round(x_y_range, decimals=Const.precision)
    #     X, Y = np.meshgrid(x_y_range, x_y_range)
    #     Z = np.zeros(shape=(len(x_y_range), len(x_y_range)))
    #     for x in range(0, len(x_y_range)):
    #         for y in range(0, len(x_y_range)):
    #             Z[x, y] = opti_func(np.array([X[x, y], Y[x, y]])[np.newaxis, :].T)
    #     return X, Y, Z

    @staticmethod
    def create_map_variables(delta, problem):
        x1 = np.arange(problem.lb, problem.ub, delta)
        x2 = np.arange(problem.lb, problem.ub, delta)
        X, Y = np.meshgrid(x1, x2)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        XY = np.vstack((X_flat, Y_flat)).T
        Z = problem.evaluate(XY)
        Z_matrix = Z.reshape(X.shape)
        return X, Y, Z_matrix
