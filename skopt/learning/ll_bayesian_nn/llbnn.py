#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:46:21 2018

@author: nimishawalgaonkar
"""
from sklearn.base import BaseEstimator, RegressorMixin

from .neural_net import NeuralNet
from .bayesian_linear_regressor import BayesianARD
from .normalization import zero_mean_unit_var_normalization

class LastLayerBayesianDeepNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state = 1,
                 normalize_input = True, normalize_output = True,
                 normalize_output_lr = False,
                 num_epochs = 50, batch_size = 50,
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
        # rng is redundant as of now.
        self.random_state = random_state
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
        
    def fit(self, X, y):
        
        """
        Fit the last layer bayesian neural network model
    
        This will consist of roughly two important tasks
        First, NN will perform the task of feature extraction 
        Second, Bayesian Linear Regressor will condition the above obtained features.
        This BLR layer is going to help us in making predictions and quantify the epistemic uncertainity
        associated with our predictions.
        
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
        
        if X.shape[0] < 50:
            
            self.n_units_3 = X.shape[0] - 2
        else:
            self.n_units_3 = self.n_units_3
        
        # Train the neural network for feature extraction
        NN = NeuralNet(X, y, self.random_state,
                                  self.normalize_input, self.normalize_output)
        
        NN.train_nn(self.num_epochs, self.batch_size,
                    self.n_units_1, self.n_units_2, self.n_units_3)
        
        self.ll_out = NN.extract_last_layer_nn_out(X)

        self.NN = NN
        
         # Bayesian Linear Regression (with Automatic Relevance determination to infer hyperparameters)
        LinearRegressor = BayesianARD(self.ll_out, y,
                                      self.normalize_output_lr,
                                      self.lr_intercept)
        m, S, sigma, alpha = LinearRegressor.train()

        self.LinearRegressor = LinearRegressor
        
        return self
        
    def retrain_NN(self, X, y):
        
        """
        Retrain NN
        """
        if X.shape[0] < 50:
            
            self.n_units_3 = X.shape[0] - 2
        else:
            self.n_units_3 = self.n_units_3
        
        # Train the neural network for feature extraction
        NN = NeuralNet(X, y, self.rng,
                                  self.normalize_input, self.normalize_output)
        
        NN.train_nn(self.num_epochs, self.batch_size,
                    self.n_units_1, self.n_units_2, self.n_units_3)
        
        self.ll_out = NN.extract_last_layer_nn_out(X)

        self.NN = NN
    
    def retrain_LR(self):
        
        """
        After the selected point (see tell()) is queried, insert the new info
        into dataset. Depending on the size of the dataset, the module decides whether
        to re-train the neural net (for feature extraction). 
        A new interpolation is then constructed.

        Keyword arguments:
        new_data -- a 1 by (m+1) array that forms the matrix [X, Y]
        """
        
        # Bayesian Linear Regression (with Automatic Relevance determination to infer hyperparameters)
        LinearRegressor = BayesianARD(self.ll_out, self.y,
                                      self.normalize_output_lr,
                                      self.lr_intercept)
        m, S, sigma, alpha = LinearRegressor.train()

        self.LinearRegressor = LinearRegressor
        
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
        if self.normalize_input:
            X = zero_mean_unit_var_normalization(X)[0]
        
        ll_out_test = self.NN.extract_last_layer_nn_out(X)
        mean, sd = self.LinearRegressor.test(ll_out_test)
        
        return mean, sd
    

    
    
        
        
        