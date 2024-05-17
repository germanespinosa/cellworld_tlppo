import random
import math
import sys
import typing
from .state import State
from .graph import Graph


class TreeNode(object):

    def __init__(self,
                 label: int,
                 state: State,
                 graph: Graph,
                 parent: "TreeNode" = None):
        self.label: int = label
        self.state: State = state
        self.graph: Graph = graph
        self.parent: TreeNode = parent
        self.children: typing.List[TreeNode] = []
        self.value: float = 0.0
        self.visits: int = 0

    def ucb1(self,
             c: float = math.sqrt(2)) -> float:
        if self.visits > 0:
            expected_reward = self.value / self.visits
        else:
            expected_reward = 0
        if c:
            if self.parent and self.parent.visits > 0:
                if self.visits > 0:
                    exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
                else:
                    exploration = sys.float_info.max
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
                best_children = [child]
            elif ucb1 == best_ucb1:
                best_children.append(child)
        return random.choice(best_children)

    def propagate_reward(self,
                         reward: float,
                         discount: float):
        self.value += reward
        self.visits += 1
        if self.parent:
            self.parent.propagate_reward(reward=reward * (1-discount),
                                         discount=discount)

    def print(self, level: int = 0):
        if level:
            print("%sL_" % ("".join([" " for i in range(level*2)])), end="")
        print(self.label)
        for child in self.children:
            child.print(level=level + 1)


class Tree(object):

    def __init__(self, graph: Graph, values: typing.Tuple[float, ...]):
        self.graph: Graph = graph
        state = State(values=values)
        self.root = TreeNode(state=state, graph=self.graph, parent=None, label=-1)
