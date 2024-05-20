import typing
import torch
from shapely import Polygon, Point
from .belief_state_component import BeliefStateComponent
from .utils import get_index, gaussian_tensor


class BeliefState(object):

    def __init__(self,
                 arena: Polygon,
                 occlusions: typing.List[Polygon],
                 definition: int,
                 components: typing.List[BeliefStateComponent],
                 other_size: int = 0):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Set device to GPU
            print("GPU is available")
        else:
            self.device = torch.device("cpu")  # Set device to CPU
            print("GPU is not available, using CPU instead")
        if other_size == 0:
            other_size = definition // 20
        self.other_size = other_size
        self.arena = arena
        self.occlusions = occlusions
        self.min_x, self.min_y, self.max_x, self.max_y = arena.bounds
        self.definition = definition
        self.granularity = (self.max_x - self.min_x) / definition
        self.y_steps = int((self.max_y - self.min_y) // self.granularity)
        self.map = torch.zeros((self.definition, self.y_steps))
        self.map.to(self.device)
        self.probability_distribution = torch.zeros(self.definition, self.y_steps)
        self.probability_distribution.to(self.device)
        self.points = [[None for _ in range(self.y_steps)] for _ in range(self.definition)]
        for i in range(self.definition):
            x = (i * self.granularity) + self.min_x
            for j in range(self.y_steps):
                y = (j * self.granularity) + self.min_y
                point = Point(x, y)
                self.points[i][j] = point
                if self.arena.contains(point):
                    for occlusion in self.occlusions:
                        if occlusion.contains(point):
                            self.map[i, j] = 0
                            self.probability_distribution[i, j] = 1.0
                            break
                    else:
                        self.map[i, j] = 1
                else:
                    self.map[i, j] = 0
        self.components = components
        for component in self.components:
            component.set_belief_state(self)
            component.on_belief_state_set()
        self.self_indices = None
        self.other_indices = None
        self.other_visible = False
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
        i, j, _, _, _, _ = self.get_location_indices(self_location)
        self.self_indices = (i, j)
        for component in self.components:
            component.on_self_location_update()

    def update_visibility(self, visibility_polygon: Polygon):
        for i in range(self.definition):
            for j in range(self.y_steps):
                if self.map[i, j]:
                    if visibility_polygon.contains(self.points[i][j]):
                        self.probability_distribution[i, j] = 0
        for component in self.components:
            component.on_visibility_update()

    def update_other_location(self, other_location: tuple):
        i, j, _, _, _, _ = self.get_location_indices(other_location)
        other_distribution = gaussian_tensor(dimensions=self.probability_distribution.shape,
                                             sigma=self.other_size,
                                             center=(i, j)).to(self.device)
        self.probability_distribution.copy_(other_distribution)
        self.probability_distribution /= self.probability_distribution.sum()
        self.other_indices = (i, j)
        self.other_visible = True
        for component in self.components:
            component.on_other_location_update()

    def tick(self):
        for component in self.components:
            component.on_tick()
            self.probability_distribution *= self.map
            self.probability_distribution /= self.probability_distribution.sum()
        self.time_step += 1
        self.other_visible = False
