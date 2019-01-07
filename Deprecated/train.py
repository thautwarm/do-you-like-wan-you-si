# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:30:25 2017

@author: misakawa
"""

from recognize.ssp import train
train(minor_epoch=50, md_name='mml2', epoch=3)
train(minor_epoch=50, md_name='mml3', epoch=3)
train(minor_epoch=50, md_name='mml4', epoch=3)
train(minor_epoch=50, md_name='mml5', epoch=3)
train(minor_epoch=50, md_name='mml6', epoch=3)