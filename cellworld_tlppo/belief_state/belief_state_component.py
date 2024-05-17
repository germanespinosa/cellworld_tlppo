import torch


class BeliefStateComponent(object):

    def __init__(self):
        self.belief_state = None

    def set_belief_state(self, belief_state: "BeliefState"):
        if self.belief_state is None:
            self.belief_state = belief_state
        else:
            raise ValueError("Belief state has already been set.")
        self.on_self_location_update()

    def on_belief_state_set(self):
        pass

    def predict(self, probability_distribution: torch.tensor):
        raise NotImplementedError

    def on_self_location_update(self):
        pass

    def on_other_location_update(self):
        pass

    def on_visibility_update(self):
        pass

    def on_tick(self):
        self.predict(self.belief_state.probability_distribution)
