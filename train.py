# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:30:25 2017

@author: misakawa
"""

from recognize.ssp import train
train(new=True, epoch=100, md_name='mml2')
train(new=True, epoch=100, md_name='mml3')
train(new=True, epoch=100, md_name='mml4')
train(new=True, epoch=100, md_name='mml5')
train(new=True, epoch=100, md_name='mml6')