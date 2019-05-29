import os
import numpy as np
import torch
from PIL import Image
from compat import imresize
from random import randint
from functools import reduce


def and_then(*fs):
    def call(x):
        return reduce(lambda a, b: b(a), fs, x)

    return call


def recursive_list(directory, format=('.png', '.jpg')):
    files = map(lambda f: '{directory}/{f}'.format(f=f, directory=directory),
                os.listdir(directory))
    res = []
    for file in files:
        if any(file.endswith(fmt) for fmt in format):
            res.append(file)
        elif os.path.isdir(file):
            res.extend(recursive_list(file))
        continue
    return res


def img_scale(img, size):
    x = imresize(img, size)
    return x


def img_redim(img):
    return np.transpose(img, (2, 0, 1))


def read_directory(dir, label):
    files = recursive_list(dir)
    return [
        and_then(
            Image.open,
            lambda x: x.convert('RGB'),
            np.array,
        )(file) for file in files
    ], np.repeat(label, len(files))


def load_model(file, **kwargs):
    return torch.load(file, **kwargs)


def dump_model(ssp, file):
    torch.save(ssp, file)
