from .belief_state_component import BeliefStateComponent
import math
from .utils import shift_tensor
import torch


class DirectedDiffusionComponent(BeliefStateComponent):
    def __init__(self, speed_rate: float):
        BeliefStateComponent.__init__(self)
        self.speed_rate = speed_rate
        self.stencil_size = None
        self.pending_time_steps = 0
        self.step_distance = None
        self.source = None
        self.step = None

    def on_belief_state_set(self):
        self.step_distance = self.speed_rate / self.belief_state.granularity

    def on_other_location_update(self):
        self.source = self.belief_state.other_indices
        target = self.belief_state.self_indices
        distance = math.dist(self.source, target)
        step_count = distance / self.step_distance
        if step_count > 0:
            self.step = tuple((ti - si) / step_count for si, ti in zip(self.source, target))
            self.pending_time_steps = int(step_count)
        else:
            self.pending_time_steps = 0

    def predict(self, probability_distribution: torch.tensor):
        if not self.belief_state.other_visible:
            if self.pending_time_steps:
                probability_distribution.copy_(shift_tensor(tensor=probability_distribution, displacement=self.step))
                self.pending_time_steps -= 1

