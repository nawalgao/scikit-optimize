#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:34:25 2018

@author: nimishawalgaonkar
"""


import numpy as np
import gaussian_process
import sys
sys.path.append('../')
from llbnn import LastLayerBayesianDeepNetRegressor


X_train = np.linspace(-1, 1, 100)[:,None]
Y_train = gaussian_process.gaussian_process(X_train)[:,0]
X_test = np.linspace(np.min(X_train), np.max(X_train), 200)[:,None]

batch_size = X_train.shape[0]

LLBNN = llbnn.LastLayerBayesianDeepNetRegressor(num_epochs = 2000, batch_size = batch_size)


