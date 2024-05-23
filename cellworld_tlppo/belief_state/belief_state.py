import typing
import torch
from shapely import Polygon, Point
from .belief_state_component import BeliefStateComponent
from .utils import get_index, gaussian_tensor
from cellworld_game import CoordinateConverter


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
            other_size = definition // 40
        self.other_size = other_size
        self.arena = arena
        self.occlusions = occlusions
        self.min_x, self.min_y, self.max_x, self.max_y = arena.bounds
        self.definition = definition
        self.granularity = (self.max_x - self.min_x) / definition
        self.y_steps = int((self.max_y - self.min_y) // self.granularity)
        self.map = torch.zeros((self.definition, self.y_steps), device=self.device)
        self.probability_distribution = torch.zeros(self.definition, self.y_steps, device=self.device)
        self.points = [[None for _ in range(self.y_steps)] for _ in range(self.definition)]
        points_list = []
        for i in range(self.definition):
            x = (i * self.granularity) + self.min_x
            for j in range(self.y_steps):
                y = (j * self.granularity) + self.min_y
                points_list.append((x,y))
                point = Point(x, y)
                self.points[i][j] = point
                if self.arena.contains(point):
                    for occlusion in self.occlusions:
                        if occlusion.contains(point):
                            self.map[i, j] = 0
                            break
                    else:
                        self.probability_distribution[i, j] = 1.0
                        self.map[i, j] = 1
                else:
                    self.map[i, j] = 0
        self.points_tensor = torch.tensor(points_list, device=self.device)

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
        self.probability_distribution /= self.probability_distribution.sum()

    def is_point(self, i, j):
        return self.map[i, j] > -1

    def get_location_indices(self, location: tuple):
        x, y = location
        i, low_i, dist_i = get_index(x, self.min_x, self.granularity)
        j, low_j, dist_j = get_index(y, self.min_y, self.granularity)
        return i, j, low_i, low_j, dist_i, dist_j

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
            other_distribution = gaussian_tensor(dimensions=self.probability_distribution.shape,
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
            for i in range(self.definition):
                for j in range(self.y_steps):
                    if self.map[i, j]:
                        if self.visibility_polygon.contains(self.points[i][j]):
                            self.probability_distribution[i, j] = 0
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

    def render(self, screen, coordinate_converter: CoordinateConverter):
        import pygame
        prob_matrix = self.probability_distribution.cpu().numpy()
        max_prob = prob_matrix.max()
        if not max_prob:
            return
        cell_size = 9

        # Get dimensions
        height, width = prob_matrix.shape
        screen_width, screen_height = width * cell_size, height * cell_size

        # Precompute color surface for efficiency
        color_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        color_surface.fill((255, 0, 0))

        for i in range(height):
            for j in range(width):
                prob = prob_matrix[i, j]
                alpha = int(255 * (prob / max_prob))
                color_surface.set_alpha(alpha)
                x, y = coordinate_converter.from_canonical((self.points[i][j].x, self.points[i][j].y))

                # Blit (copy) the colored surface onto the main screen
                screen.blit(color_surface, (x - cell_size / 2, y - cell_size / 2))

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