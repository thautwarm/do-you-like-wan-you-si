from PIL import ImageGrab
from scipy.misc import imresize
from generic_classifier.model import get_module
from generic_classifier.utils import img_redim
from pathlib import Path
from collections import deque, Counter
import numpy as np
import time
import os
import torch



def get_cls_name(cls_path):
    return cls_path[4:]

def is_cls_path(cls_path):
    return cls_path.startswith('cls_')


def get_identical_feature(shape):
    x2 = 1280
    y2 = 720 + 80
    x1 = 0
    y1 = 0
    box = (x1, y1, x2, y2)
    img = ImageGrab.grab(box)
    return img_redim(imresize(np.array(img), shape))

def predict_each(model, cascades, image_3d):
    x = model.predict_with_names(image_3d)[0]
    if x in cascades:
        sub_model, sub_cascades = cascades[x]
        x = x + '.' + predict_each(sub_model, sub_cascades, image_3d)
    return x

def is_sub_classifier_path(path):
    return path.is_dir() and any(each.is_dir() and is_cls_path(each.name) for each in path.iterdir())

def mk_cascade_classifier(path, name):
    m = get_module(str(path), str(path), (3, 240, 150), name, epoch=70, minor_epoch=7, batch_size=100, lr=0.001)
    m.use_cuda = False
    m.cpu()

    cascades = {}
    for each in path.iterdir():
        if is_sub_classifier_path(each):
            cls_name = get_cls_name(each.name)
            cascades[cls_name] = mk_cascade_classifier(each, cls_name)
    return m, cascades

def predict():
    image_3d = np.array([get_identical_feature(shape)])
    x = predict_each(root_model, root_cascades, image_3d)
    print(x)
    return x

def stable_predict(max_c=7):
    d = deque(maxlen=10)
    counter = []
    c = 0
    while True:
        for _ in range(10):
            d.append(predict())
            time.sleep(0.4)
        counter.extend(d)
        if len(set(d)) is 1:
            print('only', d)
            return d[-1]
        c += 1
        print('c = ', c)
        if c is max_c:
            counter = Counter(counter)
            [(r, _)] = counter.most_common(1)
            return r

data_path = os.path.join(os.path.split(__file__)[0], 'fgo_data')
root_model, root_cascades = mk_cascade_classifier(Path(data_path), 'mml')
shape = (240, 150)