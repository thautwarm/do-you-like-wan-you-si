# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 02:33:50 2017

@author: misakawa
"""

import os
from PIL import Image
import numpy as np
import torch
from scipy.misc import imresize
from random import randint

from functools import reduce
def and_then(*fs):
    def call(x):
        return reduce(lambda a, b: b(a), fs, x)
    return call

def recursive_list(directory, format=('.png', '.jpg') ):
    files = map(lambda f: '{directory}/{f}'.format(f=f, directory=directory), os.listdir(directory))
    res = []
    for file in files:
        if any(file.endswith(fmt) for fmt in format):
            res.append(file)
        elif os.path.isdir(file):
            res.extend(recursive_list(file))
        continue
    return res

def img_scale(img):
    a, b, *_ = img.shape
    N = randint(190, 300)
    if a > N and b > N:
        r = min(N/a, N/b)
        return imresize(img, r)
    return img

def img_redim(img):
    return np.transpose(img, (2, 0, 1))

def read_directory(dir, label):
    files = recursive_list(dir)
    return [and_then(Image.open,
                     lambda x: x.convert('RGB'),
                     np.array,
                     )(_)
            for _ in files],  np.atleast_2d(np.repeat([label], len(files))).T
            
#    return [np.transpose(np.array(Image.open(_).convert('RGB')), (2, 0, 1)) for _ in files], np.atleast_2d(np.repeat([label], len(files))).T


def load_model(file):
    import dill
    return torch.load(file, pickle_module=dill)

def dump_model(ssp, file):
    import dill
    torch.save(ssp, file, pickle_module=dill)
