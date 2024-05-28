import math
import typing
from cellworld_game import Point, Direction


class Steps:
    def __init__(self,
                 start: Point.type,
                 stops: typing.List[Point.type],
                 step_size: float = None,
                 pending: float = 0):
        self.start = start
        self.stops = stops
        self.step_size = step_size
        self.pending = pending

    def __setup__(self,
                  stop_number: int,
                  pending: float) -> typing.Tuple[bool, float, typing.Optional[tuple]]:
        if stop_number >= len(self.stops):
            return False, pending, None

        if stop_number:
            src = self.stops[stop_number - 1]
        else:
            src = self.start

        dst = self.stops[stop_number]

        distance = Point.distance(src=src, dst=dst)

        if distance + pending < self.step_size:
            start_step = dst
            pending = distance + pending
            return False, pending, None
        else:
            direction = Direction.vector(src=src, dst=dst)
            start_distance = self.step_size - pending
            distance -= start_distance
            start_step = Point.add(start=src,
                                   vector=Direction.scale_vector(vector=direction,
                                                                 scale=start_distance))

            step = Direction.scale_vector(vector=direction,
                                          scale=self.step_size)
            step_count = math.floor(distance / self.step_size)
            pending = self.step_size - (distance % self.step_size)
            return True, pending, (start_step, step, step_count)

    def __iter__(self):
        pending = self.pending
        for stop_number in range(len(self.stops)):
            has_pending_steps, pending, values = self.__setup__(stop_number=stop_number, pending=pending)
            if has_pending_steps:
                start_step, step, step_count = values
                current = start_step
                yield current
                for sn in range(step_count):
                    current = Point.add(current, step)
                    yield current
        self.pending = pending
