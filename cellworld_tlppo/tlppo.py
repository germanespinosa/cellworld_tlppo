import copy
import math
import typing

from .graph import Graph
from .mcts import Tree, TreeNode
from .belief_state import BeliefState, DirectedDiffusionComponent, GaussianDiffusionComponent


class TLPPO:
    def __init__(self,
                 graph: Graph,
                 robot_belief_state: BeliefState,
                 reward_fn: typing.Callable[[tuple, typing.Dict[str,BeliefState]], float],
                 budget: int = 100,
                 depth: int = 20,
                 ):
        gc = GaussianDiffusionComponent(2.0 / 2.34 * .25)
        dc = DirectedDiffusionComponent(.4 / 2.34 * .25)
        self.self_belief_state = BeliefState(robot_belief_state.arena,
                                             robot_belief_state.occlusions,
                                             definition=100,
                                             components=[gc, dc])
        self.robot_belief_state = robot_belief_state
        self.graph = graph
        self.budget = budget
        self.depth = depth
        self.reward_fn = reward_fn

    def get_action(self,
                   values: tuple,
                   exploration: float = math.sqrt(2),
                   discount: float = .1) -> tuple:
        belief_states_evolution = [copy.deepcopy(self.belief_states)]
        tree = Tree(graph=self.graph, values=values)
        for i in range(self.depth):
            belief_states = copy.deepcopy(belief_states_evolution[-1])
            for bs_name, belief_state in belief_states:
                belief_state.tick()
            belief_states_evolution.append(belief_states)

        for i in range(self.budget):
            node = tree.root
            reward = 0
            for d in range(self.depth):
                node = node.select(c=exploration)
                reward, _continue = self.reward_fn(node.state.values, belief_states_evolution[d])
                if not _continue:
                    break
            node.propagate_reward(reward=reward, discount=discount)

        return tree.root.select(0).state.values



