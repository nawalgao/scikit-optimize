import numpy as np
import scipy
import statsmodels.api as sm
from sklearn.linear_model import ARDRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')

from .normalization import zero_mean_unit_var_normalization
from .normalization import zero_mean_unit_var_unnormalization


class BayesianARD(object):
    def __init__(self, X, y, normalize_output = False, intercept = True):
        """
        Initialization of the linear regressor object
        Inputs:
            X : last layer output of the neural network
            y : objective function value for the corresponding set of inputs
            intercept (true or false) : whether to include intecept or not
        """
        self.X = X
        self.y = y
        self.intercept = intercept
        self.normalize_output = normalize_output

    def train(self):
        """
        Train the linear regression model based on the observed dataset
        """
        if self.normalize_output:
            (self.y,
             self.norm_mean,
             self.norm_sd) = zero_mean_unit_var_normalization(self.y)
        if self.intercept:
            train_X = sm.add_constant(self.X)
        else:
            train_X = self.X
        Phi = train_X
        regressor = ARDRegression()
        regressor.fit(Phi, self.y)
        # Best sigma
        self.sigma = np.sqrt(1. / regressor.alpha_)
        # Best alpha
        self.alpha = regressor.lambda_

        A = np.dot(Phi.T, Phi) / self.sigma ** 2. + self.alpha * np.eye(Phi.shape[1])
        A = A + np.eye(A.shape[0])*1e-5
        L = scipy.linalg.cho_factor(A)

        self.m = scipy.linalg.cho_solve(L, np.dot(Phi.T, self.y) / self.sigma ** 2)  # The posterior mean of w
        self.S = scipy.linalg.cho_solve(L, np.eye(Phi.shape[1]))           # The posterior covariance of w

        return self.m, self.S, self.sigma, self.alpha

    def test(self, X_test):
        """
        Use the trained regression parameters to estimate the objective function mean and variance
        at the new point
        Inputs:
            X_test : Design matrix at new testing points
        """
        if self.intercept:
            X_test = sm.add_constant(X_test)
        Phi_p = X_test
        Y_p = np.dot(Phi_p, self.m) # The mean prediction
        V_p_ep = np.einsum('ij,jk,ik->i', Phi_p, self.S, Phi_p) # The epistemic uncertainty
        S_p_ep = np.sqrt(V_p_ep)
        Y_l_ep = Y_p - 2. * S_p_ep  # Lower epistemic predictive bound
        
        if self.normalize_output:
            Y_p_unnorm = zero_mean_unit_var_unnormalization(Y_p, self.norm_mean, self.norm_sd)
            #S_p_ep_unnorm = normalization.zero_mean_unit_var_unnormalization(S_p_ep, self.norm_mean, self.norm_sd)
            Y_l_ep_unnorm = zero_mean_unit_var_unnormalization(Y_l_ep, self.norm_mean, self.norm_sd)
        else:
            Y_p_unnorm = Y_p
            Y_l_ep_unnorm = Y_l_ep
          
        
        S_p_unnorm = (Y_p_unnorm - Y_l_ep_unnorm)/2
        
        return Y_p_unnorm, S_p_unnorm