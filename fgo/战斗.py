import time
from fgo.common import *
from fgo.管理室 import 随便选任务
from PIL import ImageGrab
from fgo.model import stable_predict
import numpy as np

def 使用小技能(英灵位=0, 技能位=0):
    def 决定():
        def 取消():
            return MoveTo(1110, 185) + click()

        return MoveTo(875, 429) + click() + Wait(Exact(0.90)) + 取消()

    init_x = 75
    y = 580
    x = init_x + 英灵位 * 310 + 90 * 技能位
    return MoveTo(x, y) + click() + 决定()

def 普攻(slot=0):
    init_x = 150
    delta_x = 257
    y = 490
    return MoveTo(init_x + delta_x * slot, y) + click()

def 宝具(英灵位=0):
    init_x = 480
    delta_x = 230
    y = 120
    return MoveTo(init_x + delta_x * 英灵位, y) + click()

随机小技能 = []
def attack(要使用小技能=True):
    ret = Wait(Range(0.1, 0.2))
    if 要使用小技能:
        # ret += MoveTo(0, 0)
        # random.shuffle(随机小技能)

        # if not 随机小技能:
        #     随机小技能.extend([(i, j) for i in range(3) for j in range(3)])

        # i, j = 随机小技能.pop()
        # ret += 使用小技能(i, j)
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

    def 直接选牌(ctx):
        fix_dpi(attack(使用小技能=False)).eval(ctx)

    def 结束战斗(ctx):
        下一步按钮 = (1118, 679)
        ret = MoveTo(*下一步按钮) + click()
        ret += Wait(Exact(3.0))
        fix_dpi(ret).eval(ctx)

    def 取消释放界面(ctx):
        ret = Wait(Range(0.5, 1.0))
        ret += MoveTo(1145, 260)
        ret += Wait(Range(0.2, 0.3))
        ret += click()
        fix_dpi(ret).eval(ctx)

    def 不是很懂(ctx):
        Wait(Range(0.5, 1.0)).eval(ctx)

    cases = {
        'not_battle': 接任务,
        'battle_ending': 结束战斗,
        'release': 取消释放界面,
        'battle.perform.all': 战斗,
        'battle.perform.select': 直接选牌,
        'battle.wait': 不是很懂
    }

    def switch(ctx):
        kind = stable_predict()
        cases[kind](ctx)

    body = Function(switch).eval

    return Loop(Function(lambda _: True), Function(lambda ctx: body(ctx)))



