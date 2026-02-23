import logging
import torch
import numpy as np

from utils.params import UnivariateParams, BivariateParams
from utils.likelihood import torch_getLogL_bivar, torch_getLogL

logger = logging.getLogger(__name__)


def find_univariate_se(params: UnivariateParams, zvec, nvec, weights, annomat_ld, se_samples=100):
    """Compute Fisher Information Matrix and standard errors for univariate model"""
    
    # Convert inputs to tensors
    zvec = torch.as_tensor(zvec, dtype=torch.float64)
    nvec = torch.as_tensor(nvec, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    annomat_ld = torch.as_tensor(annomat_ld, dtype=torch.float64)
    
    active_params = torch.as_tensor(params.coefs[params.effective_params_mask], dtype=torch.float64)
    active_params.requires_grad = True
    active_indices = torch.where(torch.as_tensor(params.effective_params_mask, dtype=torch.bool))[0] 

    def nll_func(active_params):
        params_reconstruct = list(torch.as_tensor(params.coefs))
        for i, j in enumerate(active_indices):
            params_reconstruct[j] = active_params[i]

        return -torch_getLogL(params_reconstruct, zvec, nvec, weights, annomat_ld)
    
    # Compute Fisher Information Matrix
    fisher_info = torch.autograd.functional.hessian(nll_func, active_params)
    fisher_info = fisher_info.cpu().numpy().astype(np.float64)

    # Compute standard errors with the same error handling as bivariate
    standard_errors = np.ones_like(params.effective_params_mask, dtype=np.float64) * np.nan
    coefs_se_sample = np.tile(params.coefs, (se_samples, 1))
    try:
        cov_matrix = np.linalg.inv(fisher_info)
        _standard_errors = np.sqrt(np.diag(cov_matrix))
        standard_errors[params.effective_params_mask] = _standard_errors

        coefs_se_sample[:, params.effective_params_mask] = np.random.multivariate_normal(
            params.coefs[params.effective_params_mask], cov_matrix,
            size=se_samples, check_valid='warn', tol=1e-8)
        coefs_se_sample = np.maximum(0, coefs_se_sample)

    except RuntimeError as e:
        logger.warning(f"Could not invert Fisher Information Matrix: {e}")

    return standard_errors, coefs_se_sample


def find_bivariate_se(params: BivariateParams, zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld):
    """Compute Fisher Information Matrix and standard errors"""

    # Convert numpy arrays to PyTorch tensors with explicit float64 dtype
    zvec1 = torch.as_tensor(zvec1, dtype=torch.float64)
    zvec2 = torch.as_tensor(zvec2, dtype=torch.float64)
    nvec1 = torch.as_tensor(nvec1, dtype=torch.float64)
    nvec2 = torch.as_tensor(nvec2, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    annomat1_ld = torch.as_tensor(annomat1_ld, dtype=torch.float64)
    annomat2_ld = torch.as_tensor(annomat2_ld, dtype=torch.float64)
    annomat12_ld = torch.as_tensor(annomat12_ld, dtype=torch.float64)

    # Use params.effective_params_mask also for coefs1 and coefs2
    # (as here we're derriving SEs from bivariate likelihood, sig2_beta must be positive for both traits)
    active_params = torch.as_tensor(
        np.concat([params.coefs1[params.effective_params_mask],
                   params.coefs2[params.effective_params_mask],
                   params.rg[params.effective_params_mask]]),
        dtype=torch.float64)
    active_params.requires_grad = True

    offset0 = 0
    offset1 = offset0 + np.sum(params.effective_params_mask)
    offset2 = offset1 + np.sum(params.effective_params_mask)
    active_indices = torch.where(torch.as_tensor(params.effective_params_mask, dtype=torch.bool))[0]

    def nll_func(active_params):
        params1_reconstruct = list(torch.as_tensor(params.params1.coefs))
        params2_reconstruct = list(torch.as_tensor(params.params2.coefs))
        rho_reconstruct = list(torch.as_tensor(params.rg))

        for i, j in enumerate(active_indices):
            params1_reconstruct[j] = active_params[offset0 + i]
            params2_reconstruct[j] = active_params[offset1 + i]
            rho_reconstruct[j] = active_params[offset2 + i]

        return -torch_getLogL_bivar(params1_reconstruct, params2_reconstruct, rho_reconstruct,
                    zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld)

    # Fisher Information Matrix is the Hessian of negative log-likelihood
    fisher_info = torch.autograd.functional.hessian(nll_func, active_params)
    fisher_info = fisher_info.cpu().numpy().astype(np.float64)
    
    # Covariance matrix is inverse of Fisher Information
    standard_errors = np.ones_like(params.effective_params_mask, dtype=np.float64) * np.nan
    try:
        cov_matrix = np.linalg.inv(fisher_info)
        _standard_errors = np.sqrt(np.diag(cov_matrix))
        _standard_errors = _standard_errors[offset2:]  # only keep SE's for rg
        standard_errors[params.effective_params_mask] = _standard_errors
    except RuntimeError as e:
        logger.warning(f"Could not invert Fisher Information Matrix: {e}")

    return standard_errors
