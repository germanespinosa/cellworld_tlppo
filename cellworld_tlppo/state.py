import typing
from cellworld_game import Point


class State(object):

    def __init__(self,
                 point: typing.Tuple[float, float]):
        self.point = point


class StateList(typing.List[State]):
    def find_closest(self,
                     state: State) -> State:
        if not self:
            raise ValueError("No states found")
        min_distance = float('inf')
        closest = None
        for s in self:
            distance = Point.distance(src=s.point,
                                      dst=state.point)
            if distance < min_distance:
                closest = s
        return closest

