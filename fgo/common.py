from fgo.interops import *
import random
from functools import reduce
from copy import copy


def click():
    return reduce(Compose, [
        Wait(Range(0.15, 0.25)),
        Left(),
        Wait(Range(0.1, 0.2)),
        Left(),
        Wait(Range(0.3, 0.5))
    ])


def fix_dpi(origin: Event) -> Event:
    @origin.fmap
    def ret(event: Event):
        if isinstance(event, (MoveTo, Move)):
            event = copy(event)
            event.x = int(4 * event.x / 5)
            event.y = int(4 * (event.y + 40) / 5)
        return event

    return ret
