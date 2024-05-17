from .tlppo_learner import TlppoLearner
from .tlppo_tester import TlppoTester
from gymnasium.envs.registration import register

register(
    id='TlppoWrapper-v0',
    entry_point='cellworld_tlppo:TlppoLearner'
)

register(
    id='TlppoTester-v0',
    entry_point='cellworld_tlppo:TlppoTester'
)