import gymnasium
from gymnasium import spaces


class TlppoTester():
    def __init__(self,
                 environment_name: str,
                 lppos,
                 end_states,
                 **kwargs):
        self.environment = gymnasium.make(environment_name, **kwargs).unwrapped
        action_list = [(0.0, 0.5)] + [tuple(lppo) for lppo in lppos] + [(1.0, 0.5)]
        self.environment.action_list = action_list
        self.environment.action_space = spaces.Discrete(len(action_list))
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

    def reset(self,
              options={},
              seed=None):
        obs, info = self.environment.reset(options=options, seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.environment.step(action)
        return obs, reward, done, truncated, info

