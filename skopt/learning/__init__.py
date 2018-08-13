"""Machine learning extensions for model-based optimization."""

from .forest import RandomForestRegressor
from .forest import ExtraTreesRegressor
from .gaussian_process import GaussianProcessRegressor
from .gbrt import GradientBoostingQuantileRegressor
from .ll_bayesian_nn import LastLayerBayesianDeepNetRegressor


__all__ = ("RandomForestRegressor",
           "ExtraTreesRegressor",
           "GradientBoostingQuantileRegressor",
           "GaussianProcessRegressor",
           "LastLayerBayesianDeepNetRegressor")


