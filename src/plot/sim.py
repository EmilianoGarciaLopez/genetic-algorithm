# depreciated

import sys
import time

import matplotlib.backends.backend_agg as agg
import numpy as np
import pygame
from matplotlib import pyplot as plt
from numpy.random import rand
from pygame.locals import *


def mesh_contour(fx, bounds, delta=0.025):
    x = np.arange(bounds[0][0], bounds[0][1], delta)
    y = np.arange(bounds[1][0], bounds[1][1], delta)
    X, Y = np.meshgrid(x, y)

    @np.vectorize
    def func(x, y):
        # some arbitrary function
        return fx([x, y])

    X = X.T
    Y = Y.T
    Z = func(X, Y)
    return X, Y, Z


red = (255, 0, 0)
black = (0, 0, 0)
blue = (0, 0, 255)
white = (220, 220, 220)
gray = (200, 200, 200)


class Patch:

    def __init__(self, individual, patch_size, res, bounds, color=red, board_color=white):
        if hasattr(individual, 'position'):
            self.position = individual.position
        else:
            self.position = individual
        self.bounds = bounds
        self.res = res
        self.x = (self.position[0] - self.bounds[0][0]) / (self.bounds[0][1] - self.bounds[0][0]) * self.res
        self.y = (self.position[1] - self.bounds[1][0]) / (self.bounds[1][1] - self.bounds[1][0]) * self.res
        self.w = patch_size
        self.h = patch_size
        self.color = color
        self.board_color = board_color

    def update(self):
        self.x = (self.position[0] - self.bounds[0][0]) / (self.bounds[0][1] - self.bounds[0][0]) * self.res
        self.y = (self.position[1] - self.bounds[1][0]) / (self.bounds[1][1] - self.bounds[1][0]) * self.res

    def draw(self, surf):
        self.update()
        # self.color = self.board_color
        pygame.draw.circle(surf, self.color, center=(self.x, self.y),
                           radius=self.w / 1.5, width=0)

    def draw_rand(self, surf):
        self.update()
        r = rand()
        if r < 0.25:
            self.color = red
        elif 0.25 <= r < 0.5:
            self.color = blue
        else:
            self.color = self.board_color
        pygame.draw.circle(surf, self.color, center=(self.x, self.y),
                           radius=self.w / 2, width=0)


class Sim:

    def __init__(self, model, target, cfg, bounds, fx, name, n_levels=100, pause=0.2, patch_size=5):
        self.model = model
        self.name = name
        self.bounds = bounds
        self.floor = []
        self.fx = fx
        self.patch_size = patch_size
        self.debug = cfg['debug']
        self.m = cfg['dim']
        self.display = None
        self.animate = cfg['animate']
        self.res = self.m * self.patch_size
        self.resolution = [self.res, self.res]
        self.create_board()
        self.ini_pygame()
        self.result = []
        self.target = target
        self.n_levels = n_levels
        self.pause = pause

    def ini_pygame(self):
        if not self.debug and self.animate:
            pygame.init()
            self.display = pygame.display.set_mode(self.resolution)
            pygame.display.set_caption(self.name)

    def create_board(self):
        for particle in self.model.population:
            self.floor.append(Patch(particle, self.patch_size, self.res, self.bounds))

    def draw_board(self):
        for particle in self.model.population:
            Patch(particle, self.patch_size, self.res, self.bounds).draw(self.display)

        # for patch in self.floor:
        #     patch.draw(self.display)

    def draw_grid(self):
        for x in range(0, self.m):
            for y in range(0, self.m):
                pygame.draw.line(self.display, (230, 230, 230), (x * self.patch_size, 0),
                                 (x * self.patch_size, self.m * self.patch_size))
                pygame.draw.line(self.display, (230, 230, 230), (0, y * self.patch_size),
                                 (self.m * self.patch_size, y * self.patch_size))

    def run_sime(self, n_iter=None):
        c = 0
        # time.sleep(2)
        raw_data, size = self.countour_fig()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        while True:
            # print(self.model.best_eval, self.target, abs(self.model.best_eval - self.target))
            if n_iter is not None:
                c += 1
                if self.model.best_eval is not None:
                    if c > n_iter or abs(self.model.best_eval - self.target) < 0.0001:
                        if not self.debug and self.animate:
                            pygame.quit()
                        break
            st = time.time()
            self.step()
            time.sleep(self.pause - (time.time() - st))

            if not self.debug and self.animate:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                self.display.fill((white))
                self.display.blit(surf, (0, 0))
                self.render_gen(c, self.model.best_eval)
                # self.render_eval(self.model.best_eval)

                self.draw_board()
                # self.draw_grid()
                pygame.display.update()

    def countour_fig(self):
        delta = (self.bounds[0][1] - self.bounds[0][0]) / (self.n_levels * 10)
        X, Y, Z = mesh_contour(self.fx, self.bounds, delta=delta)
        # levels = np.arange(np.min(Z), np.max(Z), 10)
        levels = np.arange(np.min(Z), np.max(Z), (np.max(Z) - np.min(Z)) / self.n_levels)
        fig = plt.figure()
        fig.set_size_inches((self.patch_size, self.patch_size))
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        ax.invert_yaxis()
        fig.add_axes(ax)
        # fig, ax = plt.subplots(figsize=(self.patch_size, self.patch_size), dpi=self.m)
        CS = ax.contour(X, Y, Z, levels=levels)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        return raw_data, size

    def render_gen(self, c, eval):
        font = pygame.font.Font('freesansbold.ttf', 25)
        text = font.render('Gen: {}, Eval: {}'.format(c, round(eval, 4)), True, black, white)
        textRect = text.get_rect()
        textRect.center = (self.patch_size * 20, self.patch_size * 3)
        self.display.blit(text, textRect)

    def step(self):
        self.model.step()
        self.result.append(self.model.best_eval)
