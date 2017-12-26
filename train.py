# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:30:25 2017

@author: misakawa
"""

from recognize.ssp import train
train(epoch=50, md_name='mml2', max_cycle=3)
train(epoch=50, md_name='mml3', max_cycle=3)
train(epoch=50, md_name='mml4', max_cycle=3)
train(epoch=50, md_name='mml5', max_cycle=3)
train(epoch=50, md_name='mml6', max_cycle=3)