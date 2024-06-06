import typing
import cellworld_game as cg
from .belief_state_component import BeliefStateComponent
import torch


class MapComponent(BeliefStateComponent):
    def __init__(self,
                 occlusions: typing.List[cg.Polygon],):
        BeliefStateComponent.__init__(self)
        self.occlusions = occlusions
        self.valid_map = None
        self.invalid_map = None

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        self.valid_map = torch.zeros(belief_state.shape,
                               dtype=torch.bool,
                               device=belief_state.device)

        inside_arena = belief_state.arena.contains(belief_state.points)
        inside_arena_matrix = torch.reshape(inside_arena, belief_state.shape)
        self.valid_map[inside_arena_matrix] = True
        for occlusion in self.occlusions:
            inside_occlusion = occlusion.contains(belief_state.points)
            inside_occlusion_matrix = torch.reshape(inside_occlusion, belief_state.shape)
            self.valid_map[inside_occlusion_matrix] = False

        self.invalid_map = torch.logical_not(self.valid_map)

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        probability_distribution[self.invalid_map] = 0
