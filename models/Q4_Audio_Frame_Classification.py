import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from scipy.io import wavfile

from sklearn.mixture._base import  BaseMixture, _check_shape
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from scipy import linalg

from sklearn.metrics import confusion_matrix

dir_path = "/home/ece/Piyush/Coursework/PRNN/Assignment 1/archive/data/TRAIN/DR1/"

sample_rate = 16000



def _check_weights(weights, n_components):
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")

    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )

    # check normalization
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError(
            "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
            % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):

    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    return means


def _check_precision_positivity(precision, covariance_type):
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    if not (
        np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
    ):
        raise ValueError(
            "'%s precision' should be symmetric, positive-definite" % covariance_type
        )


def _check_precisions_full(precisions, covariance_type):
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    precisions = check_array(
        precisions,
        dtype=[np.float64, np.float32],
        ensure_2d=False,
        allow_nd=covariance_type == "full",
    )

    precisions_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }
    _check_shape(
        precisions, precisions_shape[covariance_type], "%s precision" % covariance_type
    )

    _check_precisions = {
        "full": _check_precisions_full,
        "tied": _check_precision_matrix,
        "diag": _check_precision_positivity,
        "spherical": _check_precision_positivity,
    }
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):

    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )

    elif covariance_type == "tied":
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        precisions = precisions_chol ** 2
        log_prob = (
            np.sum((means ** 2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X ** 2, precisions.T)
        )

    elif covariance_type == "spherical":
        precisions = precisions_chol ** 2
        log_prob = (
            np.sum(means ** 2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GaussianMixture(BaseMixture):

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _check_parameters(self, X):
        _, n_features = X.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']"
                % self.covariance_type
            )

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize(self, X, resp):
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = np.sqrt(self.precisions_init)

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )

    def aic(self, X):
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()



def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)



class GMM():
    def __init__(self, n_components, covariance_type='diag'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        # self.iter = 30

    def fit(self, X, n_iter=20, thresh=1e-2):
        self.means_ = np.zeros((self.n_components, X.shape[1]))
        self.covars_ = np.random.random((self.n_components, X.shape[1], X.shape[1]))
        self.weights_ = np.zeros(self.n_components)
        # self.precs_ = np.zeros((self.n_components, X.shape[1], X.shape[1]))
        # self.converged_ = False

        for i in range(n_iter):
            # likelihood_old = self.likelihood(X)
            # E-Step
            gamma = self.E_Step(X)
            # M-Step
            for i in range(self.n_components):
                self.weights_[i] = np.sum(gamma[:, i]) / X.shape[0]
                print(self.weights_[i])
                self.means_[i] = np.sum(self.weights_[i] * X, axis=0) / np.sum(gamma[:, i])
                x_mu = X - self.means_[i]
                self.covars_[i] = np.sum(self.weights_[i] * np.dot(x_mu.T, x_mu), axis=0) / np.sum(gamma[:, i])

            # likelihood_new = self.likelihood(X)
            # print("Iteration: {} Likelihood: {}".format(i, likelihood_new))

            # if np.abs(likelihood_new - likelihood_old) < thresh:
            #     break

    def E_Step(self, X):
        # Calculate the responsibilities
        # Calculate the log likelihood of the data
        # return the responsibilities and log likelihood
        gamma = np.zeros((X.shape[0], self.n_components))
        # print(X.shape)
        # 1/0

        for n in range(X.shape[0]):
            n_factor = 0.0
            for i in range(self.n_components):
                # print(X[n].shape)
                n_factor += self.weights_[i] * self.gaussian(X[n], self.means_[i], self.covars_[i])

            for k in range(self.n_components):
                gamma[n][k] = (self.weights_[k] * self.gaussian(X[n], self.means_[k], self.covars_[k])) / n_factor
        
        return gamma
    
    def gaussian(self, x, mu, sigma):
        # print( (x-mu).T.shape, np.linalg.pinv(sigma).shape, (x-mu).shape)
        print( np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x-mu)) , np.linalg.det(sigma))
        return np.exp(-1*0.5*np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x-mu))) / np.sqrt(2 * np.pi * np.linalg.det(sigma))

    def likelihood(self, X):
        # Calculate the log likelihood of the data
        # return the log likelihood
        llf = 0.0

        for n in range(X.shape[0]):
            n_factor = 0.0
            for i in range(self.n_components):
                n_factor += self.weights_[i] * self.gaussian(X[n], self.means_[i], self.covars_[i])

            llf += np.log(n_factor)
        
        return llf
    
    def posterior(self, x):
        prob = 0.0
        for i in range(self.n_components):
            prob += self.weights_[i] * self.gaussian(x, self.means_[i], self.covars_[i])


class MLE():
    """
    Maximum Likelihood Estimator
    """
    def __init__(self, x):
        self.x = x
        self.mu, self.sigma = self.mle(x)
    
    def mle(self, x):

        size = x[0].shape
        mu = np.zeros(size)

        for i in range(len(x)):
            mu = mu + x[i]
        mu = mu / len(x)

        sigma = np.zeros((size[0], size[0]))

        for i in range(len(x)):
            sigma = sigma + np.dot((x[i] - mu).reshape(size[0], 1), (x[i] - mu).reshape(1, size[0]))
        sigma = sigma / len(x)

        return mu, sigma
    
    def pdf(self, x):
        return np.exp(-1*0.5*np.dot(np.dot((x - self.mu).T, np.linalg.inv(self.sigma)), self.mu)) / np.sqrt(2 * np.pi * np.linalg.det(self.sigma))


class kNN():
    """
    k-Nearest Neighbors
    """
    def __init__(self, X, y, k=5):
        self.k = k
        self.X = X
        self.y = y

    def predict(self, X_test):
        """
        Predict the class of X_test
        """
        y_pred = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X)):
                distances.append(np.linalg.norm(X_test[i] - self.X[j]))
            distances = np.array(distances)
            idx = np.argsort(distances)
            idx = idx[:self.k]
            
            idx_tmp = [self.y[i] for i in idx]
            y_pred.append(max(idx_tmp, key=idx_tmp.count))
            # y_pred.append(self.y[idx])
        return y_pred


class LogisticRegression():
    """
    Logistic Regression
    """
    def __init__(self, X, y, lr=0.01, epochs=100):
        self.X = X
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.w = np.zeros(self.X.shape[1])
        self.b = 0
        self.costs = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, X, y):
        m = len(y)
        z = np.dot(X, self.w) + self.b
        h = self.sigmoid(z)
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def fit(self):
        for i in range(self.epochs):
            z = np.dot(self.X, self.w) + self.b
            h = self.sigmoid(z)
            self.w = self.w - self.lr * np.dot(self.X.T, (h - self.y))
            self.b = self.b - self.lr * np.sum(h - self.y)
            self.costs.append(self.cost(self.X, self.y))
        return self.w, self.b
    
    def predict(self, X_test):
        z = np.dot(X_test, self.w) + self.b
        h = self.sigmoid(z)
        return h


class NaiveBayes():
    """
    Naive Bayes
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.k = len(self.classes)
        self.prior = np.zeros(self.k)
        self.mu = np.zeros((self.k, X.shape[1]))
        self.sigma = np.zeros((self.k, X.shape[1], X.shape[1]))
        self.likelihood = np.zeros((self.k, X.shape[1]))
    
    def fit(self):
        for i in range(self.k):
            idx = np.where(self.y == self.classes[i])
            self.prior[i] = len(idx[0]) / len(self.y)
            self.mu[i] = np.mean(self.X[idx], axis=0)
            self.sigma[i] = np.cov(self.X[idx].T)
            self.likelihood[i] = self.normal(self.X, self.mu[i], self.sigma[i])
    
    def normal(self, X, mu, sigma):
        k, d = X.shape
        likelihood = np.zeros(k)
        for i in range(k):
            # likelihood[i] = np.exp(-0.5 * np.sum((X[i] - mu)**2 / sigma)) / (np.sqrt(2 * np.pi * sigma))
            likelihood[i] = np.exp(-1*0.5*np.dot(np.dot((x - self.mu).T, np.linalg.inv(self.sigma)), self.mu)) / np.sqrt(2 * np.pi * np.linalg.det(self.sigma))
        return likelihood
    
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            posterior = np.zeros(self.k)
            for j in range(self.k):
                posterior[j] = self.prior[j] * self.likelihood[j][i]
            y_pred.append(self.classes[np.argmax(posterior)])
        return y_pred


class PolynomialRegression():
    """
    Polynomial Regression
    """
    def __init__(self, X, y, degree=2):
        self.X = X
        self.y = y
        self.degree = degree
        self.w = np.zeros((degree + 1, X.shape[1]))
        self.b = 0
        self.costs = []
    
    def fit(self):
        for i in range(self.degree + 1):
            self.w[i] = np.dot(np.linalg.pinv(self.X), self.y)
        self.b = np.mean(self.y - np.dot(self.X, self.w))
        return self.w, self.b
    
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            y_pred.append(np.dot(self.w, X_test[i]) + self.b)
        return y_pred


class FLDClassifier():
    """
    Fisher Linear Discriminant
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.k = len(self.classes)
        self.prior = np.zeros(self.k)
        self.mean = np.zeros((self.k, X.shape[1]))
        self.cov = np.zeros((self.k, X.shape[1], X.shape[1]))
        self.w = np.zeros((self.k, X.shape[1]))
    
    def fit(self):
        for i in range(self.k):
            idx = np.where(self.y == self.classes[i])
            self.prior[i] = len(idx[0]) / len(self.y)
            self.mean[i] = np.mean(self.X[idx], axis=0)
            self.cov[i] = np.cov(self.X[idx].T)
        for i in range(self.k):
            self.w[i] = np.dot(np.linalg.pinv(self.cov[i]), (self.mean[i] - self.mean.mean(axis=0)))
        return self.w
    
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            posterior = np.zeros(self.k)
            for j in range(self.k):
                posterior[j] = self.prior[j] * np.exp(-0.5 * np.dot(np.dot((X_test[i] - self.mean[j]), np.linalg.pinv(self.cov[j])), (X_test[i] - self.mean[j]).T))
            y_pred.append(self.classes[np.argmax(posterior)])
        return y_pred


class ParzenWindow():
    """
    Parzen Window
    """
    def __init__(self, X, window_function="gaussian"):
        self.X = X
        # self.y = y
    
    def func_val_gaussian(self, x):
        val = 0.0
        for pts in self.X:
            val += np.exp(-0.5 * np.dot(x-pts, (x-pts).T)) / len(self.X)*(np.sqrt(2 * np.pi))**pts.shape[0]
        return val
    
    def posterior(self, x):
        _posterior = self.func_val_gaussian(x)
        return _posterior


def parse_wrd_timestamps(wrd_path):
    speaker_id = wrd_path.split('/')[-2]
    sentence_id = wrd_path.split('/')[-1].replace('.WRD', '')
    wrd_file = open(wrd_path)
    content = wrd_file.read()
    content = content.split('\n')
    content = [tuple(foo.split(' ') + [speaker_id, sentence_id]) for foo in content if foo != ''][1:-1]
    wrd_file.close()
    return content


def read_audio(wave_path, verbose=False):
    """
    Read Audio File
    """
    rate, data = wavfile.read(wave_path)
    assert rate == sample_rate
    return data

def extract_mfcc(x, sample_rate=44100, n_mfcc=40):
    """
    Extract MFCC from audio
    """
    mfcc = librosa.feature.mfcc(x, sr = sample_rate, n_mfcc=40)
    return mfcc

def align_data(data, words):
    aligned = []
    for tup in words:
        # print(type(data), data.shape)
        # print(tup[0], type(tup[0]))
        start = int(tup[0])
        end = int(tup[1])
        word = tup[2]
        aligned.append((data[start:end], word))
    assert len(aligned) == len(words)
    return aligned


def parse_word_waves(time_aligned_words, audio_data):
    return [align_data(data, words) for data, words in zip(audio_data, time_aligned_words)]


def gaussian_prob(x, mu, sigma):
    x_mu = x-mu
    prob = np.exp(-0.5 * np.dot(np.dot(x_mu, np.linalg.inv(sigma)), x_mu.T)) #/ (np.sqrt(2 * np.pi * np.linalg.det(sigma))**x.shape[0])
    # print(prob, np.dot(np.dot(x_mu, np.linalg.inv(sigma)), x_mu.T), np.linalg.det(sigma))
    # 1/0
    return prob

def gmm_prob(X, Mu, Sigma, Weights):
    n = Mu.shape[0]
    # print(n)

    probs = 0.0
    for i in range(n):
        probs += Weights[i] * gaussian_prob(X, Mu[i], Sigma[i])
        # print(probs)
    return probs



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)




vowels = ['a', 'e', 'i', 'o', 'u']



if __name__ == '__main__':
    print("hi")

    training_dir = "FDAW0"
    testing_dir = "MKLW0"



    wav_train = glob.glob(dir_path+training_dir+"/*.WAV.wav")
    wav_train = [ read_audio(wfile) for wfile in wav_train ]

    wav_test = glob.glob(dir_path+testing_dir+"/*.WAV.wav")
    wav_test = [ read_audio(wfile) for wfile in wav_test ]

    wrd_train = [ foo.replace('.WAV.wav', '.PHN') for foo in glob.glob(dir_path+training_dir+"/*.WAV.wav")]
    wrd_test = [ foo.replace('.WAV.wav', '.PHN') for foo in glob.glob(dir_path+testing_dir+"/*.WAV.wav")]


    wrd_tuple_train = [parse_wrd_timestamps(wrd) for wrd in wrd_train]
    wrd_tuple_test = [parse_wrd_timestamps(wrd) for wrd in wrd_test]

    train_tuple = parse_word_waves(wrd_tuple_train, wav_train)
    test_tuple = parse_word_waves(wrd_tuple_test, wav_test)

    # mn_len = min( min(bar[0].shape[0] for bar in foo for foo in train_tuple), min(bar[0].shape[0] for bar in foo for foo in test_tuple))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    mx_train, mn_train = 0, 1e10
    mx_test, mn_test = 0, 1e10

    for foo in train_tuple:
        for bar in foo:
            mx_train = max(mx_train, bar[0].shape[0])
            mn_train = min(mn_train, bar[0].shape[0])
    
    for foo in test_tuple:
        for bar in foo:
            mx_test = max(mx_test, bar[0].shape[0])
            mn_test = min(mn_test, bar[0].shape[0])
    # print(mx_train, mn_train, "\n", mx_test, mn_test)
    mn_len = min(mn_test, mn_train)

    for foo in train_tuple:
        for bar in foo:
            X_train.append(bar[0][:mn_len])
            res = [ele for ele in vowels if(ele in bar[1])]
            if bool(res):
                y_train.append(1)
            else:
                y_train.append(0)
                # print(0)
    
    for foo in test_tuple:
        for bar in foo:
            X_test.append(bar[0][:mn_len])
            res = [ele for ele in vowels if(ele in bar[1])]
            if bool(res):
                y_test.append(1)
            else:
                y_test.append(0)

    idx0 = [idx for idx, element in enumerate(y_train) if element==0]
    idx1 = [idx for idx, element in enumerate(y_train) if element==1]

    X_train0 = list(np.array(X_train)[idx0])
    X_train1 = list(np.array(X_train)[idx1])

    # print(len(X_train), len(X_train0), len(X_train1))

    """
    Maximum Likelihood Estimation based Bayes Classifier
    """
    print("\n","#"*15, "Maximum Likelihood Estimation based Bayes Classifier", "#"*15, "\n")

    mle0 = MLE(X_train0)
    mle1 = MLE(X_train1)

    acc = 0
    y_pred = []
    

    for x, y in zip(X_test, y_test):
        posterior0 = mle0.pdf(x)
        # print(posterior0)
        posterior1 = mle1.pdf(x)
        if posterior0 >= posterior1 and y == 0:
            acc += 1
            # y_pred.append(0)
        if posterior0 <= posterior1 and y == 1:
            acc += 1
            # y_pred.append(1)
        if posterior0 >= posterior1:
            y_pred.append(0)
        else:
            y_pred.append(1)

    # print("Confusion Matrix: \n", perf_measure(y_test, y_pred))
    metrics = perf_measure(y_test, y_pred)

    print("Accuracy: ", acc/len(X_test))
    print("True Positives: {} False Positives: {}".format(metrics[0], metrics[1]))
    print("True Negatives: {} False Negatives: {}".format(metrics[2], metrics[3]))
    print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
    print("Recall: ", metrics[0]/(metrics[0]+metrics[2]))
    # print()


    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    fig.savefig('cm_mle.png')

    """
    k-Nearest Neighbors Classifier
    """
    print("\n","#"*15, "k-Nearest Neighbors Classifier", "#"*15, "\n")

    for k in range(1,7):
        knn = kNN(X_train, y_train, k)
        
        y_pred = knn.predict(X_test)

        acc = 0
        for predicted, actual in zip(y_pred, y_test):
            if predicted == actual:
                acc += 1
        metrics = perf_measure(y_test, y_pred)

        # print("Accuracy: ", acc/len(X_test))
        # print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
        # print("Recall: ", metrics[0]/(metrics[0]+metrics[2]))
        print("k = {} Accuracy: {}".format(k, acc/len(y_test)))
        print("True Positives: {} False Positives: {}".format(metrics[0], metrics[1]))
        print("True Negatives: {} False Negatives: {}".format(metrics[2], metrics[3]))
        print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
        print("Recall: ", metrics[0]/(metrics[0]+metrics[2]), "\n")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
        fig.savefig('cm_knn_{}.png'.format(k))

    
    """
    Parzen based Bayes Classifier
    """
    print("\n","#"*15, "Parzen Window based Bayes Classifier", "#"*15, "\n")

    parzen0 = ParzenWindow(X_train0)
    parzen1 = ParzenWindow(X_train1)

    prior0 = y_train.count(0) / len(y_train)
    prior1 = y_train.count(1) / len(y_train)

    acc = 0
    y_pred = []

    for x, y in zip(X_test, y_test):
        posterior0 = parzen0.posterior(x)
        posterior1 = parzen1.posterior(x)

        if posterior0*prior0 >= posterior1*prior1:
            y_pred.append(0)
        else:
            y_pred.append(1)

        if posterior0*prior0 >= posterior1*prior1 and y == 0:
            acc += 1
            # y_pred.append(0)
        if posterior0*prior0 <= posterior1*prior1 and y == 1:
            acc += 1
            # y_pred.append(1)
        
    # print("Accuracy: ", acc/len(X_test))
    cm = confusion_matrix(y_test, y_pred)
    metrics = perf_measure(y_test, y_pred)

    print("Accuracy: ", (metrics[0]+metrics[2])/(metrics[0]+metrics[2]+metrics[1]+metrics[3]))
    print("True Positives: {} False Positives: {}".format(metrics[0], metrics[1]))
    print("True Negatives: {} False Negatives: {}".format(metrics[2], metrics[3]))
    # print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
    print("Recall: ", metrics[0]/(metrics[0]+metrics[2]), "\n")
    # print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
    # print("Recall: ", metrics[0]/(metrics[0]+metrics[2]))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    fig.savefig('cm_parzen.png')



    """
    Fischer Linear Discriminant Classifier
    """
    print("\n","#"*15, "Fischer Linear Discriminant Classifier", "#"*15, "\n")
    # print(np.array(X_train).shape)
    fldc = FLDClassifier(np.array(X_train), np.array(y_train))
    fldc.fit()
    y_pred = fldc.predict(np.array(X_test))

    acc = 0
    for predicted, actual in zip(y_pred, y_test):
        if predicted == actual:
            acc += 1
    # print("Accuracy: {}\n".format(acc/len(y_test)))
    cm = confusion_matrix(y_test, y_pred)

    metrics = perf_measure(y_test, y_pred)

    print("Accuracy: ", acc/len(X_test))
    print("True Positives: {} False Positives: {}".format(metrics[0], metrics[1]))
    print("True Negatives: {} False Negatives: {}".format(metrics[2], metrics[3]))
    print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
    print("Recall: ", metrics[0]/(metrics[0]+metrics[2]))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    fig.savefig('cm_flda.png')



    """
    Gaussian Mixture Model based Bayes Classifier
    """
    print("\n","#"*15, "Gaussian Mixture Model based Bayes Classifier", "#"*15, "\n")

    # gmm0 = GMM(n_components=2)
    # gmm0.fit(np.array(X_train0))

    # gmm1 = GMM(n_components=2)
    # gmm1.fit(np.array(X_train1))

    # acc = 0

    # for x, y in zip(X_test, y_test):
    #     posterior0 = gmm0.posterior(x)
    #     # print(posterior0)
    #     posterior1 = gmm1.posterior(x)
    #     if posterior0 >= posterior1 and y == 0:
    #         acc += 1
    #     if posterior0 <= posterior1 and y == 1:
    #         acc += 1
        
    # print("Accuracy: ", acc/len(X_test), "\n")
    gmm0 = GaussianMixture(n_components=2).fit(np.array(X_train0))
    gmm1 = GaussianMixture(n_components=2).fit(np.array(X_train1))

    mu0, sigma0, weights0 = gmm0.means_, gmm0.covariances_, gmm0.weights_
    mu1, sigma1, weights1 = gmm1.means_, gmm1.covariances_, gmm1.weights_

    # print(sigma0)

    # print(mu0.shape, sigma0.shape)

    acc = 0
    y_pred = []

    for x, y in zip(X_test, y_test):
        posterior0 = gmm_prob(x, mu0, sigma0, weights0)
        # print(posterior0)
        posterior1 = gmm_prob(x, mu1, sigma1, weights1)

        # y_pred = []
        # print(posterior0.shape, posterior1.shape)
        if posterior0 >= posterior1:
            y_pred.append(0)
        else:
            y_pred.append(1)

        if posterior0 >= posterior1 and y == 0:
            acc += 1
        if posterior0 <= posterior1 and y == 1:
            acc += 1
        
    # print("Accuracy: ", acc/len(X_test), "\n")
    metrics = perf_measure(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy: ", acc/len(X_test))
    print("True Positives: {} False Positives: {}".format(metrics[0], metrics[1]))
    print("True Negatives: {} False Negatives: {}".format(metrics[2], metrics[3]))
    # print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
    print("Recall: ", metrics[0]/(metrics[0]+metrics[2]))
    # print("Precision: ", metrics[0]/(metrics[0]+metrics[1]))
    # print("Recall: ", metrics[0]/(metrics[0]+metrics[2]))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    fig.savefig('cm_gmm.png')



    




    # """
    # Naive Bayes Classifier
    # """
    # print("\n","#"*15, "Naive Bayes Classifier", "#"*15, "\n")
    # # print(np.array(X_train).shape)
    # NB = NaiveBayes(np.array(X_train), np.array(y_train))
    # NB.fit()
    # y_pred = NB.predict(np.array(X_test))

    # acc = 0
    # for predicted, actual in zip(y_pred, y_test):
    #     if predicted == actual:
    #         acc += 1
    # print("Accuracy: {}\n".format(acc/len(y_test)))





    



