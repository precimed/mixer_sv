import numpy as np
import scipy.stats as stats

def likelihood_ratio_test(params_full, params_base, df_delta):
    """
    Perform likelihood ratio test
    change_log_l: difference in log-likelihood (full - nested)
    df: degrees of freedom (difference in number of parameters)
    """
    change_log_l = params_full.logL - params_base.logL
    lr_statistic = 2 * change_log_l

    if df_delta > 0:
        p_value = stats.chi2.sf(lr_statistic, df_delta)
    else:
        p_value = np.nan

    return lr_statistic, p_value
