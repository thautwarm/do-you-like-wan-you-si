import win32api, win32com, win32con, win32gui
import pythoncom
import random
import time
import abc
from typing import *

__all__ = [
    'State', 'Left', 'Right', 'Compose', 'Wait', 'Move', 'MoveTo', 'Exact',
    'Function', 'Range', 'eval_event', 'Event', 'Loop', 'If'
]


class Side:
    left = 0b01
    right = 0b10


class State:
    _x: int
    _y: int
    _active_sides: int

    @property
    def is_left_clicked(self):
        return Side.left | self._active_sides

    @property
    def is_right_clicked(self):
        return Side.right | self._active_sides

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def loc(self):
        return self._x, self._y

    @loc.setter
    def loc(self, location: Tuple[int, int]):
        win32api.SetCursorPos(location)
        self._x, self._y = location

    def left(self):
        if Side.left & self._active_sides:
            # deactivate
            op = win32con.MOUSEEVENTF_LEFTUP
        else:
            op = win32con.MOUSEEVENTF_LEFTDOWN
        win32api.mouse_event(op, *self.loc, 0, 0)
        self._active_sides ^= Side.left

    def right(self):
        if Side.right & self._active_sides:
            # deactivate
            op = win32con.MOUSEEVENTF_RIGHTUP

        else:
            op = win32con.MOUSEEVENTF_RIGHTDOWN
        self._active_sides ^= Side.right
        win32api.mouse_event(op, *self.loc, 0, 0)

    def __init__(self):
        self._x, self._y = win32api.GetCursorPos()
        self._active_sides = 0


class Event(abc.ABC):
    def eval(self, ctx: State):
        raise NotImplementedError

    def fmap(self, f):
        return f(self)

    def dump(self, prefix: str):
        raise NotImplementedError

    def __add__(self, other):
        return Compose(self, other)

    def __repr__(self):
        return self.dump('')


class Left(Event):
    def eval(self, ctx: State):
        ctx.left()

    def dump(self, _):

        return f"{_}mouse-left"


class Right(Event):
    def eval(self, ctx: State):
        ctx.right()

    def dump(self, _):
        return f"{_}mouse-right"


class Compose(Event):
    fst: Event
    then: Event

    def fmap(self, f):
        return Compose(self.fst.fmap(f), self.then.fmap(f))

    def __init__(self, fst, then):
        self.fst = fst
        self.then = then

    def dump(self, _):
        return f'{self.fst.dump(_)} ->\n{self.then.dump(_)}'

    def eval(self, ctx: State):
        self.fst.eval(ctx)
        self.then.eval(ctx)


class MoveTo(Event):
    x: int
    y: int

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dump(self, _):
        return f'{_}move-to ({self.x!r}, {self.y!r})'

    def eval(self, ctx: State):
        ctx.loc = (self.x, self.y)


class Loop(Event):
    cond: 'Function'
    do: Event

    def __init__(self, cond, do):
        self.cond = cond
        self.do = do

    def fmap(self,  f):
        return Loop(self.cond, self.do.fmap(f))

    def eval(self, ctx: State):
        cond = self.cond.eval
        do = self.do.eval
        while cond(ctx):
            do(ctx)

    def dump(self, _):
        __ = _ + '  '
        return f'{_}while \n{self.cond.dump(__)}\n{_}do\n{self.do.dump(__)}'


class If(Event):
    cond: 'Function'
    do: Event

    def __init__(self, cond, do):
        self.cond = cond
        self.do = do

    def fmap(self,  f):
        return If(self.cond, self.do.fmap(f))

    def eval(self, ctx: State):
        cond = self.cond.eval
        do = self.do.eval
        if cond(ctx):
            do(ctx)

    def dump(self, _):
        __ = _ + '  '
        return f'{_}if \n{self.cond.dump(__)}\n{_}do\n{self.do.dump(__)}'


class Move(Event):
    x: int
    y: int

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dump(self, _):
        return f'{_}move-delta ({self.x!r}, {self.y!r})'

    def eval(self, ctx: State):
        x, y = ctx.loc
        ctx.loc = (self.x + x, self.y + y)


class Time:
    pass


class Exact(Time):
    t: float

    def __init__(self, t):
        self.t = t

    def __repr__(self):
        return f'exact {self.t!r}'


class Range(Time):
    lower: float
    upper: float

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f'{self.lower!r} ~ {self.upper!r}'


class Wait(Event):
    period: Time

    def __init__(self, period):
        self.period = period

    def dump(self, _):
        return f'{_}wait {self.period} s'

    def eval(self, _: State):
        p = self.period
        if isinstance(p, Exact):
            time.sleep(p.t)
        elif isinstance(p, Range):
            lower, upper = p.lower, p.upper
            time.sleep(lower + random.random() * (upper - lower))
        else:
            raise RuntimeError


class Function(Event):
    def __init__(self, f, name=None):
        self.f = f
        self.name = name or repr(f)

    def dump(self, _):
        return f'{_}func[{self.name}]'

    def eval(self, ctx: State):
        return self.f(ctx)


def eval_event(e: Event, ctx: State = None, app_name='BlueStacks App Player'):
    ctx = ctx or State()
    pid = win32gui.FindWindow(None, app_name)
    print(win32gui.GetWindowRect(pid))
    e.eval(ctx)

