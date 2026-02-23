import scipy, scipy.sparse
from scipy.sparse import csc_matrix, issparse
import numpy as np
import mat73
import scipy.io
import os


def load_sparse_mat(path):
    mat = mat73.loadmat(path)
    return mat['LDmat'], mat['mafvec']

    
def load_ldsc_data(ldsc_mat_path):
    """
    Alternative version that returns variables separately.
    
    Returns:
    --------
    tuple
        (annomat, annomat_ld, annonames, mafvec, posvec, chrnumvec, hetvec)
    """
    
    import os
    if not os.path.exists(ldsc_mat_path):
        raise FileNotFoundError(f"LDSC file not found: {ldsc_mat_path}")
    
    # Load LDSC mat file
    mat = mat73.loadmat(ldsc_mat_path)
    # Extract the variables
    mafvec = mat['mafvec']
    posvec = mat['posvec']
    chrnumvec = mat['chrnumvec']
    hetvec = 2 * mafvec * (1 - mafvec)

    if 'annomat' in mat:
        annomat = mat['annomat']
        annonames = mat['annonames']
        annomat_ld = mat['annomat_ld'].astype(float)
        annonames = [name[0] for name in annonames]  # Convert to list of strings
    else:
        annomat = np.ones((len(mafvec), 1), dtype=float)
        annomat_ld = None
        annonames = ['base']
    return annomat, annomat_ld, annonames, mafvec, posvec, chrnumvec, hetvec

def find_weights(defvec, LD_r_squared, randprune_r2, randprune_n, disable_inverse_ld_score_weights):
    if disable_inverse_ld_score_weights:
        LD_r2_p1 = (LD_r_squared > randprune_r2)
        weights = np.zeros_like(defvec, dtype=float)
        
        for prune_i in range(randprune_n):
            randvec = np.random.rand(*defvec.shape)
            randvec[~defvec] = np.nan
            defvec_i = np.isfinite(fast_prune(randvec, LD_r2_p1))
            weights += defvec_i

        weights = weights / randprune_n
    else:
        # Inverse LD score weighting (standard LDSC approach)
        w_ld = LD_r_squared.dot(defvec.astype(int))
        weights = np.zeros_like(defvec, dtype=float)
        weights[defvec] = np.divide(1, w_ld[defvec])

    return weights

def load_trait(trait_path):
    """
    Load gwas sumstat data from a MATLAB file.
    
    Parameters:
    -----------
    trait_path : str
        Path to the trait MATLAB file
    
    Returns:
    --------
    tuple
        (zvec, nvec) - z-scores and sample sizes for the trait
    """
    
    # Check if file exists
    if not os.path.exists(trait_path):
        raise FileNotFoundError(f"Trait file not found: {trait_path}")
    
    # Load trait data
    mat = scipy.io.loadmat(trait_path)
    zvec = mat['zvec'].reshape(-1)
    nvec = mat['nvec'].reshape(-1)
    
    return zvec, nvec






def fast_prune(logpvec: np.ndarray, 
               LDmat: csc_matrix, 
               ldthresh: float = 0.2) -> np.ndarray:
    """
    Prune SNPs in high LD while retaining the most significant SNP per block.
    Optimized for CSC sparse matrices.
    
    Parameters:
    logpvec : np.ndarray
        Vector of -log10(p-values) for each SNP
    LDmat : scipy.sparse.csc_matrix
        Sparse LD matrix (r² values) in CSC format
    ldthresh : float, optional
        LD threshold for pruning (default: 0.2)
    
    Returns:
    np.ndarray : Pruned vector with retained SNPs at original values and pruned SNPs as NaN
    """
    # Validate inputs
    if not issparse(LDmat) or LDmat.format != 'csc':
        raise TypeError("LDmat must be a CSC sparse matrix")
    if len(logpvec) != LDmat.shape[0]:
        raise ValueError("logpvec length must match LDmat dimensions")
    
    # Preprocess significance values
    abs_logp = np.abs(logpvec)
    valid_mask = np.isfinite(abs_logp)
    sorted_idx = np.argsort(-abs_logp[valid_mask])  # Descending sort indices
    
    # Initialize pruning tracker
    prune_mask = np.zeros_like(logpvec, dtype=bool)
    
    # Process SNPs in significance order
    for idx in np.flatnonzero(valid_mask)[sorted_idx]:
        if prune_mask[idx]: 
            continue  # Skip already pruned SNPs
            
        # Extract LD neighborhood using CSC indices
        col_start = LDmat.indptr[idx]
        col_end = LDmat.indptr[idx+1]
        neighbor_indices = LDmat.indices[col_start:col_end]
        ld_values = LDmat.data[col_start:col_end]
        
        # Find high-LD neighbors
        high_ld_mask = (ld_values >= ldthresh)
        high_ld_neighbors = neighbor_indices[high_ld_mask]
        
        # Update pruning status
        prune_mask[high_ld_neighbors] = True
        prune_mask[idx] = False  # Retain current SNP
    
    # Apply pruning to output
    pruned_vec = logpvec.copy()
    pruned_vec[prune_mask] = np.nan
    return pruned_vec



