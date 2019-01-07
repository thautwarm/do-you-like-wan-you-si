from PIL import ImageGrab
from scipy.misc import imresize
from generic_classifier.model import get_module
from generic_classifier.utils import *
import numpy as np
import os
import torch

data_path = os.path.join(os.path.split(__file__)[0], 'fgo_data')
model = get_module(data_path, data_path, '3, 240, 150', 'mml', epoch='30', minor_epoch='5', batch_size='100', lr='0.002')
model.use_cuda = False
model.cpu()

shape = (240, 150)
def get_identical_feature():
    x2 = 1280
    y2 = 720 + 80
    x1 = 0
    y1 = 0
    box = (x1, y1, x2, y2)
    img = ImageGrab.grab(box)
    return img_redim(imresize(np.array(img), shape))


def predict():
    x = model.predict_with_names(np.array([get_identical_feature()]))[0]
    print(x)
    return x