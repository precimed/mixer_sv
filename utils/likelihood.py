import numpy as np
import torch
import math

from .params import BivariateParams
from .params import UnivariateParams

def _gaussian2_pdf(z1, z2, a11, a12, a22):
    """
    Compute the bivariate normal PDF with zero mean.

    Parameters:
    z1, z2 : np.ndarray
        Z-scores for two traits.
    a11, a12, a22 : np.ndarray
        Variance and covariance terms.

    Returns:
    np.ndarray
        PDF values.
    """
    log_pi = -np.log(2 * np.pi)
    dt = a11 * a22 - a12 ** 2  # determinant
    log_exp = -0.5 * (a22 * z1 ** 2 + a11 * z2 ** 2 - 2.0 * a12 * z1 * z2) / dt
    log_dt = -0.5 * np.log(dt)
    pdf = np.exp(log_pi + log_dt + log_exp)
    pdf = pdf + np.finfo(float).eps  # avoid zeros
    return pdf

def getLogL_bivar(zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld, params: BivariateParams):
    """
    Compute the log-likelihood for the bivariate LDSC model.

    Parameters:
    zvec1, zvec2 : np.ndarray
        Z-scores for two traits.
    nvec1, nvec2 : np.ndarray
        Sample sizes for two traits.
    weights : np.ndarray
        Weights.
    annomat_ld : np.ndarray
        LD annotation matrix.
    coefs1, coefs2, coefs12 : np.ndarray
        Model coefficients for each trait and cross-trait.

    Returns:
    float
        Log-likelihood.
    """
    defvec = np.isfinite(zvec1 + zvec2 + nvec1 + nvec2) & (weights > 0)
    nvec1 = nvec1[defvec]
    nvec2 = nvec2[defvec]
    annomat1_ld = annomat1_ld[defvec, :]
    annomat2_ld = annomat2_ld[defvec, :]
    annomat12_ld = annomat12_ld[defvec, :]
    zvec1 = zvec1[defvec]
    zvec2 = zvec2[defvec]
    weights = weights[defvec]

    coefs1 = params.coefs1
    coefs2 = params.coefs2
    coefs12 = params.coefs12

    l1 = annomat1_ld
    l2 = annomat2_ld
    l12 = annomat12_ld
    a11 = coefs1[0] + nvec1 * np.sum(l1 * coefs1[1:], axis=1)
    a22 = coefs2[0] + nvec2 * np.sum(l2 * coefs2[1:], axis=1)
    a12 = coefs12[0] + np.sqrt(nvec1 * nvec2) * np.sum(l12 * coefs12[1:], axis=1)

    phi = _gaussian2_pdf(zvec1, zvec2, a11, a12, a22)
    log_L = np.sum(weights * np.log(phi))
    params.logL = log_L  # Store log-likelihood in params
    return log_L

def getLogL(zvec, N, w, annomat_ld, params: UnivariateParams, excludevec=None):
    """
    Compute the log-likelihood for the LDSC model.

    Parameters:
    zvec : np.ndarray
        Z-scores.
    N : np.ndarray
        Sample sizes.
    w : np.ndarray
        Weights.
    annomat_ld : np.ndarray
        LD annotation matrix.
    sig2_0 : float
        Intercept variance.
    sig2_A : np.ndarray
        Annotation variances.
    excludevec : np.ndarray or None
        Boolean mask for exclusion.

    Returns:
    float
        Log-likelihood.
    """
    sig2_0 = params.coefs[0]  # Intercept variance
    sig2_A = params.coefs[1:]  # Annotation variances

    defvec = np.isfinite(zvec) & np.isfinite(N)
    if excludevec is not None:
        defvec = defvec & (~excludevec)
    N = N[defvec]
    annomat_ld = annomat_ld[defvec, :]
    zvec = zvec[defvec]
    w = w[defvec]

    l = annomat_ld  # element-wise multiplication, broadcasting
    s2 = sig2_0 + np.maximum(0, N * np.sum(l * sig2_A, axis=1))
    phi_log = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2) - (zvec ** 2) / (2 * s2)
    log_L = np.sum(w * phi_log)
    params.logL = log_L  # Store log-likelihood in params    
    return log_L

def torch_getLogL(sigma_sq, zvec, nvec, weights, annomat_ld):
    """
    Compute log-likelihood for univariate LDSC model with annotation-specific variance components.
    
    Args:
        sigma_sq: list of scalar tensors for variance components [intercept, annotations...]
        zvec: z-scores
        nvec: sample sizes  
        weights: regression weights
        annomat_ld: annotation LD matrix
    
    Returns:
        log-likelihood scalar
    """
    # Filter valid data points
    defvec = torch.isfinite(zvec) & torch.isfinite(nvec) & (weights > 0)
    zvec = zvec[defvec]
    nvec = nvec[defvec]
    weights = weights[defvec]
    l = annomat_ld[defvec, :]

    # Compute variance: s^2 = intercept + sample_size * sum(annotation_contributions)
    s2 = sigma_sq[0]
    for i, sigma_sq_i in enumerate(sigma_sq[1:]):
        s2 = nvec * l[:, i] * sigma_sq_i + s2

    # Ensure numerical stability
    s2 = torch.clamp(s2, min=torch.finfo(torch.float64).eps)
    pi = torch.tensor(math.pi, dtype=torch.float64, device=zvec.device)
    # Compute log-likelihood using univariate normal PDF
    phi_log = -0.5 * torch.log(2 * pi) - 0.5 * torch.log(s2) - (zvec ** 2) / (2 * s2)
    log_L = torch.sum(weights * phi_log)
    return log_L

def _torch_gaussian2_pdf(z1, z2, a11, a12, a22):
    """
    Compute the bivariate normal PDF with zero mean, using PyTorch.
    
    Args:
        z1, z2: z-scores for trait 1 and trait 2
        a11, a12, a22: elements of the precision matrix (inverse covariance)
    
    Returns:
        pdf: probability density values
    """
    # Ensure all constants are float64
    log_pi = torch.tensor(-math.log(2 * math.pi), dtype=torch.float64, device=z1.device)
    dt = a11 * a22 - a12 ** 2  # determinant
    dt = torch.clamp(dt, min=torch.finfo(torch.float64).eps)  # For numerical stability
    log_exp = -0.5 * (a22 * z1 ** 2 + a11 * z2 ** 2 - 2.0 * a12 * z1 * z2) / dt
    log_dt = -0.5 * torch.log(dt)
    pdf = torch.exp(log_pi + log_dt + log_exp)
    pdf = pdf + torch.finfo(torch.float64).eps  # avoid zeros
    return pdf

def torch_getLogL_bivar(sigma_sq1, sigma_sq2, rho, zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld):
    # Filter valid data points
    defvec = torch.isfinite(zvec1 + zvec2 + nvec1 + nvec2) & (weights > 0)
    zvec1 = zvec1[defvec]
    zvec2 = zvec2[defvec]
    nvec1 = nvec1[defvec]
    nvec2 = nvec2[defvec]
    weights = weights[defvec]
    l1 = annomat1_ld[defvec, :]
    l2 = annomat2_ld[defvec, :]
    l12 = annomat12_ld[defvec, :]

    # Compute covariance matrix components (all operations in float64)
    # Base component + annotation-specific contributions
    a11 = sigma_sq1[0]
    for i, sigma_sq_i in enumerate(sigma_sq1[1:]):
        a11 = nvec1 * l1[:, i] * sigma_sq_i + a11

    a22 = sigma_sq2[0]
    for i, sigma_sq_i in enumerate(sigma_sq2[1:]):
        a22 = nvec2 * l2[:, i] * sigma_sq_i + a22

    # Cross-covariance term incorporating correlations
    a12 = rho[0] * torch.sqrt(sigma_sq1[0] * sigma_sq2[0])
    for i in range(1, len(rho)):
        a12 = torch.sqrt(nvec1 * nvec2) * l12[:, i-1] * rho[i] * torch.sqrt(sigma_sq1[i] * sigma_sq2[i]) + a12
    
    # Compute likelihood using bivariate normal PDF
    phi = _torch_gaussian2_pdf(zvec1, zvec2, a11, a12, a22)
    logL = torch.sum(weights * torch.log(phi))
    return logL
