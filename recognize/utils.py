# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 02:33:50 2017

@author: misakawa
"""

import os
from PIL import Image
import numpy as np
import torch

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


def read_directory(dir, label):
    files = recursive_list(dir)
    return [np.transpose(np.array(Image.open(_).convert('RGB')), (2, 0, 1)) for _ in files], np.atleast_2d(np.repeat([label], len(files))).T


def load_model(file):
    import dill
    return torch.load(file, pickle_module=dill)

def dump_model(ssp, file):
    import dill
    torch.save(ssp, file, pickle_module=dill)
