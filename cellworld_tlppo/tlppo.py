import math
import typing
import torch
import cellworld_game as cg
from .graph import Graph
from .mcts import Tree
from .belief_state import BeliefState
from .steps import Steps
import pulsekit


class TLPPO:
    def __init__(self,
                 graph: Graph,
                 navigation: cg.Navigation,
                 robot_belief_state: BeliefState,
                 reward_fn: typing.Callable[[typing.Tuple[float, float], float], typing.Tuple[float, bool]],
                 visibility: cg.Visibility,
                 budget: int = 100,
                 depth: int = 20,
                 puff_radius: float = 0.1,
                 speed: float = .25):
        self.graph = graph
        self.navigation = navigation
        self.robot_belief_state = robot_belief_state
        self.budget = budget
        self.depth = depth
        self.speed = speed
        self.reward_fn = reward_fn
        self.visibility = visibility
        self.probability_radius = int(puff_radius * self.robot_belief_state.definition)
        self.probability_size = 2 * self.probability_radius + 1
        y, x = torch.meshgrid(torch.arange(self.probability_size, device=self.robot_belief_state.device),
                              torch.arange(self.probability_size, device=self.robot_belief_state.device),
                              indexing='ij')
        distance = torch.sqrt((x - self.probability_radius) ** 2 + (y - self.probability_radius) ** 2)
        self.probability_stencil = (distance <= self.probability_radius).float()
        self.last_action = None

    def __get_probability__(self,
                            probability_distribution: torch.Tensor,
                            location: typing.Tuple[float, float]):

        i, j, _, _, _, _ = self.robot_belief_state.get_location_indices(location)

        left_p = i - self.probability_radius
        right_p = i - self.probability_radius + self.probability_size

        top_p = j - self.probability_radius
        bottom_p = j - self.probability_radius + self.probability_size

        left_s, right_s, top_s, bottom_s = 0, self.probability_size + 1, 0, self.probability_size + 1

        if left_p < 0:
            left_s = -left_p
            left_p = 0

        if top_p < 0:
            top_s = -top_p
            top_p = 0

        if right_p >= probability_distribution.shape[0]:
            excess = 1 + right_p - probability_distribution.shape[0]
            right_s -= excess
            right_p = probability_distribution.shape[0]

        if bottom_p >= probability_distribution.shape[1]:
            excess = 1 + bottom_p - probability_distribution.shape[1]
            bottom_s -= excess
            bottom_p = probability_distribution.shape[1]

        probability_section = probability_distribution[left_p:right_p, top_p:bottom_p]
        stencil_section = self.probability_stencil[left_s:right_s, top_s:bottom_s]
        if probability_section.shape != stencil_section.shape:
            print(probability_section.shape, stencil_section.shape)
        return float((probability_section * stencil_section).sum())

    def get_action(self,
                   point: typing.Tuple[float, float],
                   exploration: float = math.sqrt(2),
                   discount: float = .1):
        # with pulsekit.CodeBlock("belief state evolution"):
        #     robot_belief_state_evolution = self.robot_belief_state.predict(num_steps=self.depth)
        with pulsekit.CodeBlock("mcts"):
            tree = Tree(graph=self.graph,
                        point=point)

            for i in range(self.budget):
                node = tree.root
                src = node.state.point
                step_count = 0
                step_remainder = 0
                iteration_reward = 0
                _continue = True
                while step_count < self.depth and _continue:
                    if self.last_action:
                        node = node.select_by_label(label=self.last_action, c=exploration)
                        self.last_action = None
                    else:
                        node = node.select(c=exploration)
                    dst = node.state.point
                    path = self.navigation.get_path(src=src,
                                                    dst=dst)
                    steps = Steps(start=src,
                                  stops=path,
                                  step_size=self.speed,
                                  pending=step_remainder)
                    for step in steps:
                        if node.visits == 0:
                            # robot_probability = self.__get_probability__(probability_distribution=robot_belief_state_evolution[step_count, :, :],
                            #                                              location=step)
                            robot_probability = self.__get_probability__(probability_distribution=self.robot_belief_state.probability_distribution,
                                                                         location=step)
                            node.step_reward, _continue = self.reward_fn(step,
                                                                         robot_probability)
                        iteration_reward += node.step_reward
                        step_count += 1
                        if not _continue or step_count == self.depth:
                            break
                    step_remainder = steps.pending
                print(step_count)
                node.propagate_reward(reward=iteration_reward,
                                      discount=discount)
        self.last_action = tree.root.select(0).label
        return tree
