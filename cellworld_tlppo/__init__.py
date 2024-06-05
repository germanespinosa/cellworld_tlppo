from .tlppo_learner import TlppoLearner
from .tlppo_tester import TlppoTester
from gymnasium.envs.registration import register
from .state import State, StateList
from .steps import Steps
from .graph import Graph
from .tlppo import TLPPO
from .mcts import Tree, TreeNode


register(
    id='TlppoWrapper-v0',
    entry_point='cellworld_tlppo:TlppoLearner'
)

register(
    id='TlppoTester-v0',
    entry_point='cellworld_tlppo:TlppoTester'
)