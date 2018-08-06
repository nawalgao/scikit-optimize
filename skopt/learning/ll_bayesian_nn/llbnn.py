#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:46:21 2018

@author: nimishawalgaonkar
"""

class LLBayesianNN(object):
    def __init__(self, rng = 1,
                 normalize_input = True, normalize_output = True,
                 normalize_output_lr = False,
                 num_epochs = 50000, batch_size = 50,
                 n_units_1 = 50, n_units_2 = 50, n_units_3 = 50,
                 lr_intercept = False):
        """
        
        Deep Neural Networks with last layer as a bayesian regressor layer
        This module performs Bayesian Linear Regression with basis function extracted
        from a feed forward neural network.
        J. Snoek paper on Scalable Bayesian Optimization
        
        Misc details used for default values :
        Number of hidden layers used : 3
        Number of nodes in each hidden layer : 
            First layer : 50
            Second layer : 50
            Third layer : nobs if nobs < 50 or 50
        
        Check out the paper for more details
        
        Parameters:
        ------------
       
        rng : int
        random seed
        
        normalize_input : True or false
        whether to normalize input features or not for fitting the neural network
        
        normalize_output : True or false
        whether to normalize outputs or not for fitting the neural network
        
        normalize_output_lr : True or false
        whether to normalize output for fitting the linear regressor
        
        num_epochs : int
        number of epochs to use for fitting the neural network
        
        batch_size : int
        number of datapoints in each batch for training the neural network
        
        n_units_1 : int
        number of units in the first hidden layer
        
        n_units_2 : int
        number of units in the second hidden layer
        
        n_units_3 : int
        number of units in the third hidden layer
        
        lr_intercept :  True or false
        whether to learn the parameter associated with linear regressor intercept or not
    
        """
    
        self.rng = rng
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.normalize_output_lr = normalize_output_lr
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Hidden layers configuration
        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        
        # Bayesian Linear Regression Configuration
        self.lr_intercept = lr_intercept
        
    def fit(self, X, Y):
        
        """
        Fit the last layer bayesian neural network model
        
        Parameters
        ----------
        * `X` [array-like, shape = (n_samples, n_features)]:
            Training data

        * `y` [array-like, shape = (n_samples, [n_output_dims])]:
            Target values

        Returns
        -------
        * `self`:
            Returns an instance of self.
        """
       
#         if X.shape[0] < 50:
#            self.n_units_3 = X.shape[0] - 2
#        else:
#            self.n_units_3 = n_units_3
    
    def predict(self, X, return_std = False):
        """
        Predict output for X.

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True)
        
        Parameters
        ----------
        * `X` [array-like, shape = (n_samples, n_features)]:
            Query points where the LLBNN model is evaluated.
            
        * `return_std` [bool, default: False]:
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        """
        
        
        
        