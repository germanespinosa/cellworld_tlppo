import typing
import math


def distance(src: typing.Tuple[float, ...],
             dst: typing.Tuple[float, ...]) -> float:
    return sum((p1 - p2) ** 2 for p1, p2 in zip(src, dst)) ** 0.5


class State(object):

    dimension_count: int = 2
    scale: typing.Tuple[float, ...] = (1.0, 1.0)
    origin:  typing.Tuple[float, ...] = (0.0, 0.0)

    def __init__(self,
                 values: typing.Tuple[float, ...]):
        if len(values) != self.dimension_count:
            raise ValueError("state size does not match: found {}, expected {}".format(len(values), self.dimension_count))
        self.values: typing.Tuple[float, ...] = values

    @property
    def magnitude(self) -> float:
        return self.distance(other=State.origin)

    def distance(self,
                 other: typing.Union["State", typing.Tuple[float, ...]]) -> float:
        if isinstance(other, State):
            other = other.values

        return distance(self.values, other)

    def atan(self,
             other: typing.Tuple[float, ...]) -> typing.Tuple[float, ...]:
        thetas: typing.List[float] = []
        for i in range(1, self.dimension_count):
            thetas.append(math.atan2(other[i] - self.values[i], other[i-1] - self.values[i-1]))
        return tuple(t for t in thetas)

    @classmethod
    def set_dimensions(cls,
                       dimension_count: int = 0,
                       scale: typing.Tuple[float, ...] = None):
        if dimension_count == 0 and scale is None:
            raise ValueError("either dimension_count or scale must be specified")
        if dimension_count == 0:
            dimension_count = len(scale)
        if scale is None:
            scale = [1.0 for i in range(dimension_count)]
        cls.dimension_count = dimension_count
        cls.scale = scale
        cls.origin = tuple(0.0 for i in range(dimension_count))

    def __add__(self, other: typing.Union["State", typing.Tuple[float, ...]]) -> "State":
        if isinstance(other, State):
            other = other.values
        values = tuple(i + j for i, j in zip(self.values, other))
        return State(values=values)

    def __iadd__(self, other: typing.Union["State", typing.Tuple[float, ...]]) -> "State":
        if isinstance(other, State):
            other = other.values
        self.values = tuple(i + j for i, j in zip(self.values, other))
        return self

    def __sub__(self, other: typing.Union["State", typing.Tuple[float,...]]) -> "State":
        if isinstance(other, State):
            other = other.values
        values = tuple(i - j for i, j in zip(self.values, other))
        return State(values=values)

    def __neg__(self) -> "State":
        return State(values=tuple(-x for x in self.values))

    def __mul__(self, other: float) -> "State":
        return State(values=tuple(x * other for x in self.values))

    def normalize(self) -> "State":
        return State(values=tuple(x / self.magnitude for x in self.values))

    def direction(self, other: typing.Union["State", typing.Tuple[float, ...]]) -> typing.Tuple[float, ...]:
        if isinstance(other, State):
            other = other.values
        diff = -(self - other)
        return diff.normalize().values

    def move(self, direction: typing.Tuple[float, ...], distance: float) -> "State":
        return State(values=tuple(x + d * distance for x, d in zip(self.values, direction)))


class StateList(typing.List[State]):
    def find_closest(self,
                     state: State) -> State:
        if not self:
            raise ValueError("No states found")
        min_distance = float('inf')
        closest = None
        for s in self:
            distance = s.distance(other=state.values)
            if distance < min_distance:
                closest = s
        return closest
