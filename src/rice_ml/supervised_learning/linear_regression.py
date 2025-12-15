import numpy as np

from rice_ml.utilities._validation import *


class LinearRegression:
    """
    Ordinary least squares Linear Regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if fit_intercept=False.
    n_features_in : int
        Number of features seen during fit.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X = _validate_2d_array(X)
        y = _validate_1d_array(y)
        _check_same_length(X, y, "X", "y")
        _check_finite_if_numeric(X)
        _check_finite_if_numeric(y)

        self.n_features_in = X.shape[1]

        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.intercept_ = 0.0
            self.coef_ = beta

        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Returns predicted values.
        """

        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fit yet.")
        
        X = _validate_2d_array(X)
        if X.shape[1] != self.n_features_in:
            raise ValueError("Input should have the same number of features as the fitted X.")
        
        y_pred = np.dot(X, self.coef_) + self.intercept_

        return y_pred
        

    def score(self, X, y):
        """
        Return the coefficient of determination (R^2) of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        float
            R^2 score. Best possible score is 1.0.
        """
        
        y = _validate_1d_array(y)
        y_pred = self.predict(X)

        rss = np.sum((y-y_pred)**2)
        tss = np.sum((y-y.mean())**2)

        if tss == 0:
            return 1.0 if rss > 0 else 0.0

        return 1 - (rss / tss)