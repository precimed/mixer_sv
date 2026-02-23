import numpy as np
import scipy.stats as stats


class UnivariateParams():
    def __init__(self, coefs):
        self.coefs = coefs
        self.logL = np.nan

        eps = 100 * np.finfo(float).eps
        self.effective_params_mask = self.coefs > eps

    @property
    def num_effective_params(self):
        return np.sum(self.effective_params_mask)

    # AIC calculation: by definition, AIC = 2k - 2ln(L), where k is the number of parameters and L is the likelihood.
    # https://en.wikipedia.org/wiki/Akaike_information_criterion
    @property
    def AIC(self):
        return 2 * self.num_effective_params - 2 * self.logL

    __repr__ = lambda self: f"UnivariateParams(coefs={self.coefs}, logL={self.logL})"

def _calc_rg_from_coefs(coefs1, coefs2, coefs12, effective_params_mask):
    max_rho = 0.999
    idx = effective_params_mask
    rg = np.zeros_like(coefs12)
    rg[idx] = coefs12[idx] / np.sqrt(coefs1[idx] * coefs2[idx])
    rg = np.maximum(np.minimum(rg, max_rho), -max_rho)
    return rg


class BivariateParams():
    def __init__(self, coefs1, coefs2, coefs12=None, rho=None):
        self.params1 = UnivariateParams(coefs1)
        self.params2 = UnivariateParams(coefs2)
        self.effective_params_mask = self.params1.effective_params_mask & self.params2.effective_params_mask

        if coefs12 is None and rho is not None:
            assert(np.all(rho >= -1) and np.all(rho <= 1)), "Correlation coefficient must be in [-1, 1]"
            self.coefs12 = rho * np.sqrt(coefs1 * coefs2)
        elif coefs12 is not None:
            self.coefs12 = _calc_rg_from_coefs(coefs1, coefs2, coefs12, self.effective_params_mask) * np.sqrt(coefs1 * coefs2)
        else:
            raise ValueError("Either coefs12 or rho must be provided.")

        self.logL = np.nan

    @property
    def coefs1(self):
        return self.params1.coefs

    @property
    def coefs2(self):
        return self.params2.coefs

    @property
    def rg(self):
        return _calc_rg_from_coefs(self.coefs1, self.coefs2, self.coefs12, self.effective_params_mask)

    @property
    def num_effective_params(self):
        return np.sum(self.effective_params_mask) + self.params1.num_effective_params + self.params2.num_effective_params

    @property
    def AIC(self):
        return 2 * self.num_effective_params - 2 * self.logL

    __repr__ = lambda self: f"BivariateParams(coefs1={self.coefs1}, coefs2={self.coefs2}, rg={self.rg}, logL={self.logL})"
