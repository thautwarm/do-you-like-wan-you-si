from api import Mouse, get_loc, win32api
from time import sleep
from random import random
from PIL import ImageGrab
import numpy as np
from recognize.ssp import load_model
from matplotlib import pyplot
pyplot.interactive(True)
_model1 = load_model('recognize\\mml2')
_model2 = load_model('recognize\\mml6')
_model5 = load_model('recognize\\mml5')
_model3 = load_model('recognize\\mml3')
_model4 = load_model('recognize\\mml4')
def model(datas):
    r1 = _model1.predict(datas)
    r2 = _model2.predict(datas)
    r3 = _model4.predict(datas)
    r4 = _model3.predict(datas)
    r5 = _model1.predict(datas)
    return np.round((r1 + r2 + r3 + r4 + r5)/5).astype(np.int8)



def click():
        sleep(0.5*random() + 0.3)
        Mouse.left_down()
        sleep(0.01*random() + 0.05)
        Mouse.left_up()

def 扫二维码(app_name='BlueStacks App Player'):
    sleep(0.2*random()+0.5)
    Mouse.left_down()
    sleep(0.2*random()+2)
    Mouse.left_up()
    sleep(0.1*random()+0.1)
    left, top, right, bottom = get_loc(app_name)
    win32api.SetCursorPos([(left + right) // 2, bottom - 100])
    def new_recognized():
        sleep(random() + 0.3)
        Mouse.left_down()
        sleep(random() + 0.1)
        Mouse.left_up()

    def close():
        sleep(random()+0.1)
        win32api.SetCursorPos([left+20, top + 90])
        click()
        sleep(random()+0.1)

    new_recognized()
    close()

def 找二维码(app_name='BlueStacks App Player'):
    left, top, right, bottom = get_loc(app_name)
    def get_pics(vertical_split_pixel: 'in my screen, this value fits best' = 100):

        height_from = top + 150
        height_to = bottom + 150
        img = np.array(ImageGrab.grab((left, height_from, right - 130, height_to)))

        last = 0
        for i in range(0, np.shape(img)[0] - 1, vertical_split_pixel):
            if last is i:
                continue
            click_where = left + 100, height_from + 50 + last
            print('-> ', last, i)
#            pyplot.figure()
#            pyplot.imshow(img[last:i, :, :])
            yield click_where, np.transpose(img[last:i, :, :], (2, 0, 1))
            last = i
    click_loc_list, datas = zip(*get_pics())
    targets = model(datas)
    print(targets)
    ids = np.arange(len(targets))[targets==1]
    if any(ids):
        print('found')
        i = ids[-1]
        # 扫最后一个二维码(lastest one
        print(click_loc_list[i])
        win32api.SetCursorPos(click_loc_list[i])
        click()
        扫二维码()

sleep(10)
找二维码()
