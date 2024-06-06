import random
import math
import sys
import typing
import cellworld_game as cg
from .graph import Graph
from .state import State
import math


class TreeNode(object):

    def __init__(self,
                 label: int,
                 state: State,
                 graph: Graph,
                 parent: "TreeNode" = None):
        self.label: int = label
        self.state = state
        self.graph: Graph = graph
        self.parent: TreeNode = parent
        self.children: typing.List[TreeNode] = []
        self.value: float = -math.inf
        self.visits: int = 0
        self.step_reward: float = 0
        self.remaining_step: float = 0
        self.data = None

    def ucb1(self,
             c: float = math.sqrt(2)) -> float:
        if self.visits > 0:
            expected_reward = self.value
        else:
            expected_reward = 0
        if c:
            if self.parent and self.parent.visits > 0:
                if self.visits > 0:
                    exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
                else:
                    exploration = math.inf
            else:
                exploration = 0
        else:
            exploration = 0
        return expected_reward + exploration

    def expand(self):
        connections = self.graph.edges[self.label]
        for connection in connections:
            child_node = self.graph.nodes[connection]
            child = TreeNode(label=connection,
                             state=child_node.state,
                             graph=self.graph,
                             parent=self)
            self.children.append(child)

    def select_by_label(self,
                        label: int,
                        c: float) -> "TreeNode":

        if not self.children:
            self.expand()

        for child in self.children:
            if child.label == label:
                return child

        return self.select(c=c)

    def select(self,
               c: float) -> "TreeNode":

        if not self.children:
            self.expand()

        if not self.children:
            return self

        best_ucb1 = self.children[0].ucb1(c=c)
        best_children = [self.children[0]]
        for child in self.children[1:]:
            ucb1 = child.ucb1(c=c)
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_children = [child]
            elif ucb1 == best_ucb1:
                best_children.append(child)
        selected = random.choice(best_children)
        return selected

    def get_best(self):
        best_value = None
        best_child = None
        for child in self.children:
            if child.visits == 0:
                continue
            if best_value is None:
                best_value = child.value
                best_child = child
            else:
                if child.value > best_value:
                    best_value = child.value
                    best_child = child
        return best_child

    def propagate_reward(self,
                         reward: float,
                         discount: float):
        self.visits += 1
        self.value = max(reward, self.value)
        if self.parent:
            self.parent.propagate_reward(reward=self.value * (1-discount),
                                         discount=discount)

    def print(self, level: int = 0):
        if level:
            print("%sL_" % ("".join([" " for i in range(level*2)])), end="")
        print(self.label)
        for child in self.children:
            child.print(level=level + 1)


class Tree(object):

    def __init__(self,
                 graph: Graph,
                 point: cg.Point.type):
        self.graph: Graph = graph
        state = State(point=point)
        self.root = TreeNode(state=state,
                             graph=self.graph,
                             parent=None,
                             label=-1)
        closest_node = self.graph.get_nearest(point=point)
        for label in self.graph.edges[closest_node.label]:
            node = self.graph.nodes[label]
            self.root.children.append(TreeNode(graph=graph,
                                               label=node.label,
                                               state=node.state,
                                               parent=self.root))
