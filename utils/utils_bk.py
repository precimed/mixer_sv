import numpy as np

eps = np.finfo(float).eps

def find_fold_enrich(annomat, sig2_beta_i_base, sig2_beta_i_full, h2_selector=lambda x: x[0]):
    # default h2_selector() works under assumption that the first annotation is 'base' (all SNPs)
    h2_base_ROI = annomat.T @ sig2_beta_i_base
    h2_full_ROI = annomat.T @ sig2_beta_i_full
    h2_base_total = h2_selector(h2_base_ROI)
    h2_full_total = h2_selector(h2_full_ROI)
    fold_enrich = np.ones_like(h2_full_ROI) * np.nan
    if (h2_full_total > eps) and (h2_base_total > eps):
        idx = h2_base_ROI > eps
        fold_enrich[idx] = (h2_full_ROI[idx] / h2_full_total) / (h2_base_ROI[idx] / h2_base_total)
    return fold_enrich, h2_full_total


def find_fold_enrich_se(annomat_ROI, annomat, sig2_beta_i, coefs, coefs_se_sample):
    # Calculate heritability contributions and enrichments, and their SE's
    fold_enrich, h2_total = find_fold_enrich(annomat_ROI, sig2_beta_i, sig2_beta_i * (annomat @ coefs[1:]))

    h2_total_se_sample = np.zeros((coefs_se_sample.shape[0], ))
    fold_enrich_se_sample = np.zeros((coefs_se_sample.shape[0], annomat_ROI.shape[1] ))
    for i in range(coefs_se_sample.shape[0]):
        fold_enrich_se_sample[i, :], h2_total_se_sample[i] = find_fold_enrich(annomat_ROI, sig2_beta_i,
            sig2_beta_i * (annomat @ coefs_se_sample[[i], 1:].ravel()))

    h2_total_se = np.std(h2_total_se_sample) if (h2_total > eps) else np.nan

    fold_enrich_se = np.zeros((fold_enrich_se_sample.shape[1], )) * np.nan
    for i in range(fold_enrich_se_sample.shape[1]):
        vals = fold_enrich_se_sample[:, i]
        vals = vals[np.isfinite(vals)]
        fold_enrich_se[i] = np.std(vals) if vals.size > 1 else np.nan

    return fold_enrich, fold_enrich_se, h2_total, h2_total_se
