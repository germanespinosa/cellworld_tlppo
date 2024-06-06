import typing

import cellworld_game
import torch
from .belief_state_component import BeliefStateComponent
from .utils import get_index
from cellworld_game import CoordinateConverter, Polygon


class BeliefState(object):

    def __init__(self,
                 arena: Polygon,
                 definition: int,
                 components: typing.List[BeliefStateComponent]):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Set device to GPU
        else:
            self.device = torch.device("cpu")  # Set device to CPU
        self.rendered = False
        self.arena = arena
        self.min_x, self.min_y, self.max_x, self.max_y = arena.bounds()
        self.definition = definition
        self.granularity = (self.max_x - self.min_x) / definition
        self.y_steps = int((self.max_y - self.min_y) // self.granularity)
        self.shape = (self.y_steps, self.definition)
        self.size = self.y_steps * self.definition
        self.points = torch.zeros((self.size, 2),
                                  dtype=torch.float32,
                                  device=self.device)
        index = 0
        for j in range(self.y_steps):
            y = (j * self.granularity) + self.min_y + self.granularity / 2
            for i in range(self.definition):
                x = (i * self.granularity) + self.min_x + self.granularity / 2
                self.points[index][0] = x
                self.points[index][1] = y
                index += 1

        self.components = components
        for component in self.components:
            component.set_belief_state(self)
            component.on_belief_state_set(belief_state=self)
        self.self_indices = None
        self.other_indices = None
        self.self_location = None
        self.other_location = None
        self.other_visible = False
        self.visibility_polygon = None
        self.time_step = 0
        self.probability_distribution = torch.ones(self.shape,
                                                   dtype=torch.float32,
                                                   device=self.device)
        self.probability_distribution /= self.probability_distribution.sum()
        self.j_grid, self.i_grid = torch.meshgrid(torch.arange(self.shape[1], device=self.device),
                                                  torch.arange(self.shape[0], device=self.device), indexing='xy')

    def get_mask_in_radius(self,
                           point: cellworld_game.Point.type,
                           radius: float) -> torch.Tensor:

        i, j, _, _, _, _ = self.get_location_indices(point)
        max_distance = int(radius * self.definition)
        distance = torch.sqrt((self.i_grid - i) ** 2 + (self.j_grid - j) ** 2)
        within_radius = distance <= max_distance
        return within_radius

    def get_probability_in_radius(self,
                                  point: cellworld_game.Point.type,
                                  radius: float,
                                  probability_distribution: torch.Tensor = None) -> float:
        if probability_distribution is None:
            probability_distribution = self.probability_distribution

        within_radius = self.get_mask_in_radius(point=point, radius=radius)
        return float((probability_distribution * within_radius).sum())

    def get_mask_in_distance_to_segment(self,
                                        src: cellworld_game.Point.type,
                                        dst: cellworld_game.Point.type,
                                        distance: float):
        i0, j0, _, _, _, _ = self.get_location_indices(src)
        i1, j1, _, _, _, _ = self.get_location_indices(dst)
        max_distance = int(distance * self.definition)
        pi = i1 - i0
        pj = j1 - j0
        norm = pi * pi + pj * pj
        u = ((self.i_grid - i0) * pi + (self.j_grid - j0) * pj) / norm
        u = torch.clamp(u, 0, 1)
        x_closest = i0 + u * pi
        y_closest = j0 + u * pj
        dist = torch.sqrt((self.i_grid - x_closest) ** 2 + (self.j_grid - y_closest) ** 2)
        mask = dist <= max_distance
        return mask

    def get_probability_in_distance_to_segment(self,
                                               src: cellworld_game.Point.type,
                                               dst: cellworld_game.Point.type,
                                               distance: float,
                                               probability_distribution: torch.Tensor = None):

        if probability_distribution is None:
            probability_distribution = self.probability_distribution

        within_distance = self.get_mask_in_distance_to_segment(src=src, dst=dst, distance=distance)
        probability = probability_distribution * within_distance

        # import matplotlib.pyplot as plt
        # plt.imshow(within_distance.cpu().numpy(), cmap='viridis')  # Convert to NumPy for plotting
        # plt.colorbar()
        # plt.title("2D Gaussian Distributed Tensor")
        # plt.show()
        # plt.close()
        #
        # plt.imshow(probability_distribution.cpu().numpy(), cmap='viridis')  # Convert to NumPy for plotting
        # plt.colorbar()
        # plt.title("2D Gaussian Distributed Tensor")
        # plt.show()
        # plt.close()
        #
        # plt.imshow(probability.cpu().numpy(), cmap='viridis')  # Convert to NumPy for plotting
        # plt.colorbar()
        # plt.title("2D Gaussian Distributed Tensor")
        # plt.show()
        # plt.close()

        return float(probability.sum())

    def reset(self):
        self.time_step = 0
        self.probability_distribution[:, :] = 1
        self.probability_distribution /= self.probability_distribution.sum()
        for component in self.components:
            component.on_reset()
    def get_location_indices(self, location: tuple):
        x, y = location
        i, low_i, dist_i = get_index(x, self.min_x, self.granularity)
        j, low_j, dist_j = get_index(y, self.min_y, self.granularity)
        return j, i, low_j, low_i, dist_j, dist_i

    def update_self_location(self, self_location: tuple):
            self.self_location = self_location

    def update_visibility(self, visibility_polygon: Polygon):
        self.visibility_polygon = visibility_polygon

    def update_other_location(self, other_location: tuple):
        self.other_location = other_location

    def tick(self):
        if self.self_location:
            i, j, _, _, _, _ = self.get_location_indices(self.self_location)
            self.self_indices = (i, j)
            for component in self.components:
                component.on_self_location_update(self.self_location,
                                                  self.self_indices,
                                                  self.time_step)

        if self.other_location:
            i, j, _, _, _, _ = self.get_location_indices(self.other_location)
            self.other_indices = (i, j)
            self.other_visible = True
            for component in self.components:
                component.on_other_location_update(self.other_location,
                                                   self.other_indices,
                                                   self.time_step)
        elif self.visibility_polygon:
            for component in self.components:
                component.on_visibility_update(self.visibility_polygon,
                                               self.time_step)

        for component in self.components:
            component.on_tick(self.probability_distribution,
                              self.time_step)
            total_sum = self.probability_distribution.sum()
            if total_sum > 0:
                self.probability_distribution /= total_sum

        self.time_step += 1
        self.other_visible = False
        self.visibility_polygon = None
        self.other_location = None
        self.self_location = None
        self.rendered = False

    def render(self, screen, coordinate_converter: CoordinateConverter):
        import pygame
        # Create a surface capable of handling alpha
        values = (self.probability_distribution * 255 / self.probability_distribution.max()).int().cpu().numpy()
        heatmap_surface = pygame.Surface(values.shape[::-1], pygame.SRCALPHA)
        pix_array = pygame.PixelArray(heatmap_surface)
        for y in range(values.shape[1]):
            for x in range(values.shape[0]):
                value = values[values.shape[0] - x - 1, y]
                pix_array[y, x] = (255, 0, 0, value)

        # Delete the pixel array to unlock the surface
        del pix_array

        # Scale the surface to the window size
        scaled_heatmap = pygame.transform.scale(heatmap_surface,
                                                size=(coordinate_converter.screen_width,
                                                      coordinate_converter.screen_height))
        screen.blit(scaled_heatmap, (0, 0))

    def predict(self, num_steps: int) -> torch.Tensor:
        predictions = torch.zeros((num_steps, self.shape[0], self.shape[1]),
                                  device=self.device,
                                  dtype=torch.float32)
        probability_distribution = self.probability_distribution
        for i in range(num_steps):
            time_step = self.time_step + 1
            predictions[i, :, :] = probability_distribution
            probability_distribution = predictions[i, :, :]
            for component in self.components:
                component.predict(probability_distribution=probability_distribution,
                                  time_step=time_step)
                probability_distribution /= probability_distribution.sum()
        return predictions