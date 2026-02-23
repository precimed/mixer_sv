import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
import torch
from .params import UnivariateParams, BivariateParams
from .likelihood import getLogL_bivar, getLogL


def wols(X, y, w):
    model = LinearRegression()
    model.fit(X, y, sample_weight=w)
    intercept = np.atleast_1d(model.intercept_).ravel()
    coef = model.coef_.ravel()
    return np.concat([intercept, coef])




def nnls_weighted(X, y, w):
    # Add intercept column
    X_ = np.hstack([np.ones((X.shape[0], 1)), X])
    # Apply sqrt of weights
    sqrt_w = np.sqrt(w)
    Xw = X_ * sqrt_w[:, np.newaxis]
    yw = y * sqrt_w
    # Fit NNLS
    coef, _ = nnls(Xw, yw)
    return coef  # [intercept, coef1, coef2, ...]


def train_mixer_univariate_MoM(zvec, nvec, weights, annomat_ld):
    defvec = np.isfinite(zvec + nvec) & (weights > 0)

    # Trait 1 model: E[z₁²] = σ₀² + N₁ × l × σ_all²
    X = annomat_ld * nvec[:, None]
    y = zvec**2
    X_sub = X[defvec, :]
    y_sub = y[defvec]
    w_sub = weights[defvec]
    coefs1 = nnls_weighted(X_sub, y_sub, w_sub)  # [σ₀², σ_all²] for trait 1
    params = UnivariateParams(coefs1)
    getLogL(zvec, nvec, weights, annomat_ld, params)
    return params


def train_mixer_bivariate_MoM(zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld):
    defvec = np.isfinite(zvec1 + zvec2 + nvec1 + nvec2) & (weights > 0)

    # Trait 1 model: E[z₁²] = σ₀² + N₁ × l × σ_all²
    X = annomat1_ld * nvec1[:, None]
    y = zvec1**2
    X_sub = X[defvec, :]
    y_sub = y[defvec]
    w_sub = weights[defvec]
    coefs1 = nnls_weighted(X_sub, y_sub, w_sub)  # [σ₀², σ_all²] for trait 1

    # Trait 2 model: E[z₂²] = σ₀² + N₂ × l × σ_all²  
    X = annomat2_ld * nvec2[:, None]
    y = zvec2**2
    X_sub = X[defvec, :]
    y_sub = y[defvec]
    w_sub = weights[defvec]
    coefs2 = nnls_weighted(X_sub, y_sub, w_sub)  # [σ₀², σ_all²] for trait 2

    # Cross-trait model: E[z₁z₂] = √(N₁N₂) × l × σ₁₂_all
    X = np.sqrt(nvec1.flatten() * nvec2.flatten())
    X = annomat12_ld * X[:, None]
    y = zvec1 * zvec2
    X_sub = X[defvec, :]
    y_sub = y[defvec]
    w_sub = weights[defvec]
    coeffs12 = wols(X_sub, y_sub, w_sub)  # [cov₀, cov_all] cross-trait

    params = BivariateParams(coefs1, coefs2, coeffs12)
    getLogL_bivar(zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld, params)
    getLogL(zvec1, nvec1, weights, annomat1_ld, params.params1)
    getLogL(zvec2, nvec2, weights, annomat2_ld, params.params2)
    return params
