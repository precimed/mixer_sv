import numpy as np
import torch
import logging
from utils.likelihood import getLogL_bivar, getLogL
from utils.likelihood import torch_getLogL_bivar, torch_getLogL
from utils.params import UnivariateParams, BivariateParams

logger = logging.getLogger(__name__)

# =====================================================================================
# PYTORCH IMPLEMENTATION FOR MAXIMUM LIKELIHOOD OPTIMIZATION
# =====================================================================================

class MixerUnivariateModel(torch.nn.Module):
    """
    PyTorch implementation of univariate MiXeR model for maximum likelihood optimization.
    
    This model parameterizes the univariate normal likelihood using:
    - sigma_sq: variance components [intercept, annotation1, annotation2, ...]
    
    The model supports partitioned annotations (e.g., within/outside genomic regions).
    """
    
    def __init__(self, params: UnivariateParams):
        super().__init__()
        self.orig_params = params
        self.sigma_sq_raw = torch.nn.ParameterList()
        eps = np.finfo(float).eps

        for i in range(len(params.coefs)):
            coef_value = params.coefs[i]
            do_optimize = params.effective_params_mask[i]
            if do_optimize:
                coef_value = np.log(max(coef_value, eps))

            # Use log parameterization to ensure positivity
            p = torch.nn.Parameter(torch.tensor(coef_value, dtype=torch.float64))
            p.requires_grad = True if do_optimize else False
            self.sigma_sq_raw.append(p)
    
    def forward(self, zvec, nvec, weights, annomat_ld):
        """
        Forward pass: compute negative log-likelihood for optimization.
        
        Args:
            zvec: z-scores
            nvec: sample sizes
            weights: regression weights
            annomat_ld: LD-weighted annotation matrix
        
        Returns:
            neg_logL: negative log-likelihood (for minimization)
        """
        # Transform parameters back to original space (maintaining positivity)
        sigma_sq = [(torch.exp(param) if param.requires_grad else param) 
                    for param in self.sigma_sq_raw]
        logL = torch_getLogL(sigma_sq, zvec, nvec, weights, annomat_ld)
        return -logL  # Return negative for minimization
    
    @torch.no_grad()
    def extract_params(self):
        """Extract optimized parameters for downstream analysis."""
        sigma_sq = [(torch.exp(param) if param.requires_grad else param).cpu().numpy().astype(np.float64) 
                    for param in self.sigma_sq_raw]
        params = UnivariateParams(np.array(sigma_sq))
        params.effective_params_mask = self.orig_params.effective_params_mask
        return params


class MixerBivariateModel(torch.nn.Module):
    """
    PyTorch implementation of bivariate MiXeR model for maximum likelihood optimization.
    
    This model parameterizes the bivariate normal likelihood using:
    - sigma_sq1, sigma_sq2: variance components for each trait
    - rho: correlation coefficients between traits
    
    The model supports partitioned annotations (e.g., within/outside genomic regions).
    """

    def __init__(self, params: BivariateParams):
        super().__init__()

        self.orig_params = params
        rho_values = params.rg
        self.rho_raw = torch.nn.ParameterList()
        for i in range(len(rho_values)):
            do_optimize = params.effective_params_mask[i]
            p = torch.nn.Parameter(torch.tensor(np.arctanh(rho_values[i]), dtype=torch.float64))
            p.requires_grad = True if do_optimize else False
            self.rho_raw.append(p)

    def forward(self, zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld):
        """
        Forward pass: compute negative log-likelihood for optimization.
        
        Args:
            zvec1, zvec2: z-scores for traits 1 and 2
            nvec1, nvec2: sample sizes for traits 1 and 2
            weights: regression weights
            annomat_ld: LD-weighted annotation matrix
        
        Returns:
            neg_logL: negative log-likelihood (for minimization)
        """
        # Transform parameters back to original space (maintaining float64)
        rho = [torch.tanh(rho) for rho in self.rho_raw]
        
        logL = torch_getLogL_bivar(
            list(torch.as_tensor(self.orig_params.coefs1)), list(torch.as_tensor(self.orig_params.coefs2)),
            rho,
            zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld
        )

        return -logL  # Return negative for minimization
    
    @torch.no_grad()
    def extract_params(self):
        """Extract optimized parameters for downstream analysis."""
        rho = [torch.tanh(rho).cpu().numpy().astype(np.float64) for rho in self.rho_raw]
        params = BivariateParams(self.orig_params.coefs1, self.orig_params.coefs2, rho=np.array(rho))
        params.effective_params_mask = self.orig_params.effective_params_mask
        return params


def train_mixer_univariate_MLE(
    zvec_np, nvec_np, weights_np, annomat_ld_np, 
    params: UnivariateParams,
    n_epochs=500, lr=0.01, device="cpu", grad_clip_norm=5.0,
    convergence_tol=0.01, patience=50, out_loss_history=None
):
    """
    Train the univariate MiXeR model using PyTorch optimization.
    
    Args:
        zvec_np: z-scores (numpy array)
        nvec_np: sample sizes (numpy array) 
        weights_np: regression weights (numpy array)
        annomat_ld_np: LD-weighted annotation matrix (numpy array)
        params: initial variance component estimates
        n_epochs: maximum number of training epochs
        lr: learning rate
        device: torch device ('cpu' or 'cuda')
        grad_clip_norm: gradient clipping threshold
        convergence_tol: convergence tolerance
        patience: early stopping patience
        out_loss_history: output variable to store loss history (optional)
    Returns:
        optimized_params: optimized variance components
    """
    
    # Convert numpy arrays to PyTorch tensors with explicit float64 dtype
    zvec = torch.tensor(zvec_np, dtype=torch.float64, device=device)
    nvec = torch.tensor(nvec_np, dtype=torch.float64, device=device)
    weights = torch.tensor(weights_np, dtype=torch.float64, device=device)
    annomat_ld = torch.tensor(annomat_ld_np, dtype=torch.float64, device=device)
    
    # Initialize model and optimizer
    model = MixerUnivariateModel(params).to(device)
    model = model.double()  # Ensure float64 precision
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with convergence monitoring
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = model(zvec, nvec, weights, annomat_ld)
        
        # Check for numerical issues
        if not torch.isfinite(loss):
            logger.warning(f"\tStopping early at epoch {epoch}: loss is NaN or Inf!")
            break
        
        # Convergence check with early stopping
        if abs(best_loss - loss.item()) < convergence_tol:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\tConverged at epoch {epoch}")
                break
        else:
            best_loss = min(best_loss, loss.item())
            patience_counter = 0
        
        # Backpropagation and parameter update
        loss.backward()
        
        # Optional gradient clipping for stability
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        if out_loss_history is not None:
            out_loss_history.append(loss.item())
        
        # Periodic evaluation and logging
        if (logging.INFO >= logging.root.level) and (epoch % 25 == 0 or epoch == n_epochs - 1):
            params = model.extract_params()
            params.logL = -loss.item()
            logger.info(f"\tEpoch {epoch:3d}: logL = {params.logL:.6f}")
    
    # Extract optimized parameters
    params = model.extract_params()
    params.logL = getLogL(zvec_np, nvec_np, weights_np, annomat_ld_np, params)
    return params


def train_mixer_bivariate_MLE(
    zvec1_np, zvec2_np, nvec1_np, nvec2_np, weights_np, annomat1_ld_np, annomat2_ld_np, annomat12_ld_np,
    params: BivariateParams,
    n_epochs=500, lr=0.01, device="cpu", grad_clip_norm=5.0,
    convergence_tol=0.01, patience=50, out_loss_history=None
):
    """
    Train the bivariate MiXeR model using PyTorch optimization.
    
    This function implements maximum likelihood estimation with:
    - Early stopping based on convergence criteria
    - Gradient clipping for numerical stability
    - Periodic evaluation and logging
    
    Args:
        zvec1_np, zvec2_np: z-scores (numpy arrays)
        nvec1_np, nvec2_np: sample sizes (numpy arrays)
        weights_np: regression weights (numpy array)
        annomat_ld_np: LD-weighted annotation matrix (numpy array)
        coefs1, coefs2, coefs12: initial parameter estimates from Method of Moments
        n_epochs: maximum number of training epochs
        lr: learning rate
        device: torch device ('cpu' or 'cuda')
        grad_clip_norm: gradient clipping threshold
        convergence_tol: convergence tolerance
        patience: early stopping patience
        out_loss_history: output variable to store loss history (optional)
    
    Returns:
        model: trained PyTorch model
    """
    
    # Convert numpy arrays to PyTorch tensors with explicit float64 dtype
    zvec1 = torch.tensor(zvec1_np, dtype=torch.float64, device=device)
    zvec2 = torch.tensor(zvec2_np, dtype=torch.float64, device=device)
    nvec1 = torch.tensor(nvec1_np, dtype=torch.float64, device=device)
    nvec2 = torch.tensor(nvec2_np, dtype=torch.float64, device=device)
    weights = torch.tensor(weights_np, dtype=torch.float64, device=device)
    annomat1_ld = torch.tensor(annomat1_ld_np, dtype=torch.float64, device=device)
    annomat2_ld = torch.tensor(annomat2_ld_np, dtype=torch.float64, device=device)
    annomat12_ld = torch.tensor(annomat12_ld_np, dtype=torch.float64, device=device)

    # Initialize model and optimizer
    model = MixerBivariateModel(params).to(device)

    # Ensure model parameters are float64
    model = model.double()  # This converts all parameters to float64
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with convergence monitoring
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = model(zvec1, zvec2, nvec1, nvec2, weights, annomat1_ld, annomat2_ld, annomat12_ld)

        # Check for numerical issues
        if not torch.isfinite(loss):
            logger.warning(f"\tStopping early at epoch {epoch}: loss is NaN or Inf!")
            break
        
        # Convergence check with early stopping
        if abs(best_loss - loss.item()) < convergence_tol:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\tConverged at epoch {epoch}")
                break
        else:
            best_loss = min(best_loss, loss.item())
            patience_counter = 0
        
        # Backpropagation and parameter update
        loss.backward()
        
        # Optional gradient clipping for stability
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        if  out_loss_history is not None:
            out_loss_history.append(loss.item())
        
        # Periodic evaluation and logging
        if (logging.INFO >= logging.root.level) and (epoch % 25 == 0 or epoch == n_epochs - 1):
            params = model.extract_params()
            params.logL = -loss.item()
            logger.info(f"\tEpoch {epoch:3d}: logL = {params.logL:.6f}")

    # Extract optimized parameters
    params = model.extract_params()
    getLogL_bivar(zvec1_np, zvec2_np, nvec1_np, nvec2_np, weights_np, annomat1_ld_np, annomat2_ld_np, annomat12_ld_np, params)
    getLogL(zvec1_np, nvec1_np, weights_np, annomat1_ld_np, params.params1)
    getLogL(zvec2_np, nvec2_np, weights_np, annomat2_ld_np, params.params2)
    return params
