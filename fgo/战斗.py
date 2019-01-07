import time
from fgo.common import *
from fgo.管理室 import 随便选任务
from PIL import ImageGrab
from collections import deque, Counter
from fgo.model import predict
import numpy as np

def vote():
    return predict()

def stable_vote():
    d = deque(maxlen=10)
    counter = []
    c = 0
    while True:
        for _ in range(10):
            d.append(vote())
            time.sleep(0.4)
        counter.extend(d)
        if len(set(d)) is 1:
            print('only', d)
            return d[-1]
        c += 1
        print('c = ', c)
        if c is 8:
            counter = Counter(counter)
            [(r, _)] = counter.most_common(1)
            return r

def 使用小技能(英灵位=0, 技能位=0):
    def 决定():
        def 取消():
            return MoveTo(1110, 185) + click()

        return MoveTo(875, 429) + click() + Wait(Exact(0.90)) + 取消()

    init_x = 75
    y = 580
    x = init_x + 英灵位 * 310 + 90 * 技能位
    return MoveTo(x, y) + click() + 决定()


随机小技能 = [(i, j) for i in range(3) for j in range(3)]
def attack():
    def 宝具(英灵位=0):
        init_x = 480
        delta_x = 230
        y = 120
        return MoveTo(init_x + delta_x * 英灵位, y) + click()

    def 普攻(slot=0):
        init_x = 150
        delta_x = 257
        y = 490
        return MoveTo(init_x + delta_x * slot, y) + click()

    ret = MoveTo(0, 0)
    random.shuffle(随机小技能)
    for i, j in 随机小技能[:2]:
        ret += 使用小技能(i, j)
    ret += MoveTo(1132, 608) + click()
    ret += Wait(Exact(1.5)) # 等一下，不然放不了第一个宝具
    ret += reduce(Compose, (宝具(i) for i in range(3)))
    l = [(普攻, i) for i in range(5)]
    random.shuffle(l)
    l = map(lambda a: a[0](a[1]), l[:3])
    ret += reduce(Compose, l)
    ret += Wait(Exact(7.5))
    return ret


def 贪玩():

    def 接任务(ctx):
        fix_dpi(Wait(Exact(5)) + 随便选任务() + Wait(Exact(12))).eval(ctx)

    def 战斗(ctx):
        fix_dpi(attack()).eval(ctx)

    def 结束战斗(ctx):
        下一步按钮 = (1118, 679)
        ret = MoveTo(*下一步按钮) + click()
        ret += Wait(Exact(3.0))
        fix_dpi(ret).eval(ctx)

    cases = {
        'not_battle': 接任务,
        'battle': 战斗,
        'battle_ending': 结束战斗
    }

    def switch(ctx):
        kind = stable_vote()
        cases[kind](ctx)

    body = Function(switch).eval

    return Loop(Function(lambda _: True), Function(lambda ctx: body(ctx)))



