#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:36:05 2018

@author: nimishawalgaonkar
"""

import numpy as np
import gaussian_process
import sys
sys.path.append('../')
import llbnn



X_train = np.linspace(-1, 1, 100)[:,None]
Y_train = gaussian_process.gaussian_process(X_train)[:,0]
X_test = np.linspace(np.min(X_train), np.max(X_train), 200)[:,None]


batch_size = X_train.shape[0]

LLBNN = llbnn.LastLayerBayesianDeepNetRegressor(num_epochs = 2000, batch_size = batch_size)
LLBNN.fit(X_train, Y_train)
print('Training is done')

# Testing on training data itself
r = LLBNN.predict(X_test)
#print ('Mean')
#print (r[0])
#print ('SD')
#print (r[1].shape)
#print ('Lower')
#print (r[3].shape)
#print ('Upper')
#print (r[2].shape)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')
plt.figure()
plt.plot(X_test, r[0])
r1 = r[0] + 2*r[1]
r2 = r[0] - 2*r[1]
plt.fill_between(X_test.flatten(), r1, r2, color=sns.color_palette()[2], alpha=0.25)
plt.scatter(X_train, Y_train, color = 'r')
