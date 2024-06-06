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
        self.puff_radius = puff_radius
        self.visibility = visibility
        self.last_action = None

    def get_action(self,
                   point: typing.Tuple[float, float],
                   exploration: float = math.sqrt(2),
                   discount: float = .1,
                   previous_tree: Tree = None):

        tree = Tree(graph=self.graph,
                    point=point)

        # evaluates previous plan

        robot_belief_state_evolution = self.robot_belief_state.predict(self.depth)

        for i in range(self.budget):
            node = tree.root
            src = node.state.point
            step_count = 0
            step_remainder = 0
            iteration_reward = 0
            _continue = True
            node_steps = []
            node_steps_rewards = []
            while step_count < self.depth and _continue:
                node = node.select(c=exploration)
                dst = node.state.point
                path = self.navigation.get_path(src=src,
                                                dst=dst)
                steps = Steps(start=src,
                              stops=path,
                              step_size=self.speed,
                              pending=step_remainder)
                for step in steps:
                    robot_probability = self.robot_belief_state.get_probability_in_radius(point=step,
                                                                                          radius=self.puff_radius)
                    step_reward, _continue = self.reward_fn(step,
                                                            robot_probability)
                    if iteration_reward > step_reward:
                        iteration_reward = step_reward
                    step_count += 1
                    node_steps.append(step)
                    node_steps_rewards.append(step_reward)
                    if not _continue or step_count == self.depth:
                        break
                step_remainder = steps.pending
            node.propagate_reward(reward=iteration_reward,
                                  discount=discount)
            node.data = (node_steps, node_steps_rewards)
        self.last_action = tree.root.select(0).label
        return tree
