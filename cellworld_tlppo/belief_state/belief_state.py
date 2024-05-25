import math
import typing
import torch
from .belief_state_component import BeliefStateComponent
from .utils import get_index, gaussian_tensor
from cellworld_game import CoordinateConverter, Polygon


class BeliefState(object):

    def __init__(self,
                 arena: Polygon,
                 occlusions: typing.List[Polygon],
                 definition: int,
                 components: typing.List[BeliefStateComponent],
                 other_size: int = 0):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Set device to GPU
        else:
            self.device = torch.device("cpu")  # Set device to CPU
        if other_size == 0:
            other_size = max(definition // 40, 2)
        self.rendered = False
        self.other_size = other_size
        self.arena = arena
        self.occlusions = occlusions
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

        self.map = torch.zeros(self.shape,
                               device=self.device,
                               dtype=torch.float32)

        inside_arena = self.arena.contains(self.points)
        inside_arena_matrix = torch.reshape(inside_arena, self.shape)
        self.map[inside_arena_matrix] = 1

        for occlusion in self.occlusions:
            inside_occlusion = occlusion.contains(self.points)
            inside_occlusion_matrix = torch.reshape(inside_occlusion, self.shape)
            self.map[inside_occlusion_matrix] = 0

        self.components = components
        for component in self.components:
            component.set_belief_state(self)
            component.on_belief_state_set()
        self.self_indices = None
        self.other_indices = None
        self.self_location = None
        self.other_location = None
        self.other_visible = False
        self.visibility_polygon = None
        self.time_step = 0
        self.probability_distribution = self.map.clone()
        self.probability_distribution /= self.probability_distribution.sum()

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
                component.on_self_location_update()

        if self.other_location:
            i, j, _, _, _, _ = self.get_location_indices(self.other_location)
            other_distribution = gaussian_tensor(dimensions=self.shape,
                                                 sigma=self.other_size,
                                                 center=(i, j),
                                                 device=self.device)
            self.probability_distribution.copy_(other_distribution)

            self.probability_distribution *= self.map
            self.probability_distribution /= self.probability_distribution.sum()
            self.other_indices = (i, j)
            self.other_visible = True
            for component in self.components:
                component.on_other_location_update()
        elif self.visibility_polygon:
            in_view = self.visibility_polygon.contains(self.points)
            in_view_matrix = torch.reshape(in_view, self.shape)
            self.probability_distribution[in_view_matrix] = 0
            self.probability_distribution /= self.probability_distribution.sum()
            for component in self.components:
                component.on_visibility_update()

        for component in self.components:
            component.on_tick()
            self.probability_distribution *= self.map
            self.probability_distribution /= self.probability_distribution.sum()

        self.time_step += 1
        self.other_visible = False
        self.visibility_polygon = None
        self.other_location = None
        self.self_location = None
        self.rendered = False

    def get_probability(self, location: tuple, radius: float):
        i, j, low_i, low_j, dist_i, dist_j = self.get_location_indices(location)
        r = int(radius * self.definition)

        size = 2 * r + 1
        center = r  # The center index

        # Create a grid of indices
        y, x = torch.meshgrid(torch.arange(size, device=self.device),
                              torch.arange(size, device=self.device),
                              indexing='ij')

        # Compute the distance from the center for each index
        distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)

        # Create the tensor with values 1 where the distance is less than or equal to r
        stencil = (distance <= r).float()

        return float((self.probability_distribution[i - r: i - r + size, j - r: j - r + size] * stencil).sum())

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

