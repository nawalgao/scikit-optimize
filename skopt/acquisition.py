import numpy as np
import warnings

from scipy.stats import norm

from rpy2.robjects.packages import STAP
with open('qEI_call.r', 'r') as f:
    string = f.read()
qEI = STAP(string, "qEI_call")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as rpyn
rcpp = importr("Rcpp")
rcpp.sourceCpp('qEI.cpp')


def gaussian_acquisition_1D(X, model, y_opt=None, acq_func="LCB",
                            acq_func_kwargs=None, return_grad=True):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    return _gaussian_acquisition(np.expand_dims(X, axis=0),
                                 model, y_opt, acq_func=acq_func,
                                 acq_func_kwargs=acq_func_kwargs,
                                 return_grad=return_grad)


def _gaussian_acquisition(X, model, y_opt=None, acq_func="LCB",
                          return_grad=False, acq_func_kwargs=None):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X is {}-dimensional, however,"
                         " it must be 2-dimensional.".format(X.ndim))

    if acq_func_kwargs is None:
        acq_func_kwargs = dict()
    xi = acq_func_kwargs.get("xi", 0.01)
    kappa = acq_func_kwargs.get("kappa", 1.96)

    # Evaluate acquisition function
    per_second = acq_func.endswith("ps")
    if per_second:
        model, time_model = model.estimators_

    if acq_func == "LCB":
        func_and_grad = gaussian_lcb(X, model, kappa, return_grad)
        if return_grad:
            acq_vals, acq_grad = func_and_grad
        else:
            acq_vals = func_and_grad
    
    elif acq_func in ["EI", "PI", "EIps", "PIps"]:
        if acq_func in ["EI", "EIps"]:
            func_and_grad = gaussian_ei(X, model, y_opt, xi, return_grad)
        else:
            func_and_grad = gaussian_pi(X, model, y_opt, xi, return_grad)

        if return_grad:
            acq_vals = -func_and_grad[0]
            acq_grad = -func_and_grad[1]
        else:
            acq_vals = -func_and_grad

        if acq_func in ["EIps", "PIps"]:

            if return_grad:
                mu, std, mu_grad, std_grad = time_model.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=True)
            else:
                mu, std = time_model.predict(X, return_std=True)

            # acq = acq / E(t)
            inv_t = np.exp(-mu + 0.5*std**2)
            acq_vals *= inv_t

            # grad = d(acq_func) * inv_t + (acq_vals *d(inv_t))
            # inv_t = exp(g)
            # d(inv_t) = inv_t * grad(g)
            # d(inv_t) = inv_t * (-mu_grad + std * std_grad)
            if return_grad:
                acq_grad *= inv_t
                acq_grad += acq_vals * (-mu_grad + std*std_grad)

    else:
        raise ValueError("Acquisition function not implemented.")

    if return_grad:
        return acq_vals, acq_grad
    return acq_vals


def approx_qei(X, model, maxima, x_pending = None,
               num_sampled_points = 5,
               num_batches_eval = 400,
               strategy_batch_selection = 'random'):
    """
    Use Mickael Binois approximation to qEI function
    This is used to calculate qEI score for batches 
    
    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.
        
     * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.
    
     Returns
    -------
    * `values`: [array-like, shape=(len(num_batches_eval),)]:
        qEI values for each batch
    """
    # Converting x_pending list to numpy array
    if x_pending is not None:  
        x_pending = np.array(x_pending)
    
    batches = []
    cc_vec = np.zeros(num_batches_eval)
    # Batch preparation
    for i in range(num_batches_eval):   
        if strategy_batch_selection == 'random':
            rel_ind = np.random.choice(X.shape[0], num_sampled_points, replace=False)
            b = X[rel_ind,:]
            if x_pending is not None:
                b = np.vstack([x_pending, b])
        else:
            ValueError ("No such sampling strategy exists ..")
        batches.append(b)
        mean, covar = model.predict(b, return_cov=True)
        cc = qEI.qEI_approx(mean, covar, maxima)
        cc_num = rpyn.ri2py(cc)
        cc_vec[i] = cc_num
    max_qEI_val = np.max(cc_vec)
    max_qEI_val_ind = np.argmax(cc_vec)
    best_batch = batches[max_qEI_val_ind]
    
    return best_batch, batches, cc_vec, max_qEI_val
    
    


def gaussian_lcb(X, model, kappa=1.96, return_grad=False):
    """
    Use the lower confidence bound to estimate the acquisition
    values.

    The trade-off between exploitation and exploration is left to
    be controlled by the user through the parameter ``kappa``.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `kappa`: [float, default 1.96 or 'inf']:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        If set to 'inf', the acquisition function will only use the variance
        which is useful in a pure exploration setting.
        Useless if ``method`` is set to "LCB".

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.

    * `grad`: [array-like, shape=(n_samples, n_features)]:
        Gradient at X.
    """
    # Compute posterior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

            if kappa == "inf":
                return -std, -std_grad
            return mu - kappa * std, mu_grad - kappa * std_grad

        else:
            mu, std = model.predict(X, return_std=True)
            if kappa == "inf":
                return -std
            return mu - kappa * std


def gaussian_pi(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    """
    Use the probability of improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)`form a gaussian with a
    certain mean and standard deviation approximated by the model.

    The PI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 1``, if ``f(x) < y_opt`` and ``u(f(x)) = 0``,
    if``f(x) > y_opt``.

    This means that the PI condition does not care about how "better" the
    predictions are than the previous values, since it gives an equal reward
    to all of them.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `y_opt` [float, default 0]:
        Previous minimum value which we would like to improve upon.

    * `xi`: [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)
        else:
            mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    values[mask] = norm.cdf(scaled)

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std**2

        return values, improve_grad * norm.pdf(scaled)

    return values


def gaussian_ei(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    """
    Use the expected improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)`form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `y_opt` [float, default 0]:
        Previous minimum value which we would like to improve upon.

    * `xi`: [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

        else:
            mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std ** 2
        cdf_grad = improve_grad * pdf
        pdf_grad = -improve * cdf_grad
        exploit_grad = -mu_grad * cdf - pdf_grad
        explore_grad = std_grad * pdf + pdf_grad

        grad = exploit_grad + explore_grad
        return values, grad

    return values
