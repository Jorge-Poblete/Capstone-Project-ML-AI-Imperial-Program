# ================================================================
#FULL STRATEGY MODULE
# ================================================================

import numpy as np

from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture

# ✅ Correct framework imports
from src.utils.sampling import (
    sobol_sampling,
    polar_sampling,
    dirichlet_sampling,
    simplex_sampling,
    anisotropic_sampling,
    rf_mean_std
)

# ================================================================
# STRATEGY SELECTOR
# ================================================================
class StrategySelector:

    @staticmethod
    def execute(strategy_name, gp, X, y, params, dim, seed):

        mapping = {
            "INITIAL": InitialStrategy,
            "EI_LOCAL": EILocalStrategy,
            "MULTI_LEVEL": MultiLevelStrategy,
            "REFINEMENT": RefinementStrategy,
            "DOUBLE_RADIUS": DoubleRadiusStrategy,
            "TURBO": TurboStrategy,
            "RF_EI": RandomForestStrategy,

            "TPE": TPEStrategy,
            "ADAPTIVE_MIX": AdaptiveMixStrategy,
            "DIRICHLET": DirichletStrategy,
            "SIMPLEX": SimplexAdaptiveStrategy,
            "ANISOTROPIC": AnisotropicStrategy,
            "TURBO_RF_EI": TurboRFStrategy,
            "LOCAL_GLOBAL_GP": LocalGlobalStrategy,
        }

        strategy = mapping.get(strategy_name, EILocalStrategy)

        return strategy.run(gp, X, y, params, dim, seed)


# ================================================================
# EXPECTED IMPROVEMENT
# ================================================================

def expected_improvement(mean, std, y_best, xi=0.01):

    std = np.maximum(std, 1e-12)

    improvement = mean - y_best - xi
    Z = improvement / std

    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std < 1e-9] = 0

    return ei


# ================================================================
# INITIAL STRATEGY
# ================================================================

class InitialStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        n = params.get("n_cand", 10000)

        candidates = sobol_sampling(n, dim, seed)

        return candidates[0], None, "INITIAL"


# ================================================================
# EI LOCAL
# ================================================================

class EILocalStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        radius = params.get("r_local", 0.1)
        center = X[np.argmax(y)]

        candidates = polar_sampling(center, radius, 5000, dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], center, "EI_LOCAL"


# ================================================================
# MULTI LEVEL
# ================================================================

class MultiLevelStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        global_n = params.get("global", 5000)
        meso_n = params.get("meso", 3000)
        local_n = params.get("local", 2000)

        best = X[np.argmax(y)]

        global_c = sobol_sampling(global_n, dim, seed)
        meso_c = polar_sampling(best, 0.2, meso_n, dim, seed)
        local_c = polar_sampling(best, 0.05, local_n, dim, seed)

        candidates = np.vstack([global_c, meso_c, local_c])

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "MULTI_LEVEL"


# ================================================================
# REFINEMENT
# ================================================================

class RefinementStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        radius = params.get("r_local", 0.05)
        best = X[np.argmax(y)]

        candidates = polar_sampling(best, radius, 5000, dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "REFINEMENT"


# ================================================================
# DOUBLE RADIUS
# ================================================================

class DoubleRadiusStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        best = X[np.argmax(y)]

        small = polar_sampling(best, params.get("r_small", 0.05), 3000, dim, seed)
        big = polar_sampling(best, params.get("r_big", 0.2), 3000, dim, seed)

        candidates = np.vstack([small, big])

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "DOUBLE_RADIUS"


# ================================================================
# TURBO
# ================================================================

class TurboStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        center = X[np.argmax(y)]
        radius = params.get("tr_init", 0.1)

        low = np.maximum(0, center - radius)
        high = np.minimum(1, center + radius)

        candidates = np.random.uniform(low, high, (5000, dim))

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], center, "TURBO"


# ================================================================
# RANDOM FOREST EI
# ================================================================

class RandomForestStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        rf = RandomForestRegressor(n_estimators=300, random_state=seed)
        rf.fit(X, y)

        candidates = sobol_sampling(params.get("global", 20000), dim, seed)

        mu, std = rf_mean_std(rf, candidates)

        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], X[np.argmax(y)], "RF_EI"


# ================================================================
# TPE STRATEGY
# ================================================================

class TPEStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        n = params.get("n_cand", 20000)
        candidates = sobol_sampling(n, dim, seed)

        threshold = np.quantile(y, 0.8)

        good = X[y >= threshold]
        bad = X[y < threshold]

        if len(good) < 5 or len(bad) < 5:
            score = np.ones(n)
        else:
            gmm_good = GaussianMixture(2).fit(good)
            gmm_bad = GaussianMixture(2).fit(bad)

            lg = np.exp(gmm_good.score_samples(candidates))
            lb = np.exp(gmm_bad.score_samples(candidates))

            score = lg / (lb + 1e-12)

        return candidates[np.argmax(score)], X[np.argmax(y)], "TPE"


# ================================================================
# ADAPTIVE MIX
# ================================================================

class AdaptiveMixStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed):

        rf = RandomForestRegressor(n_estimators=300, random_state=seed)
        rf.fit(X, y)

        candidates = sobol_sampling(params.get("n_cand", 20000), dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        rf_mu = rf.predict(candidates)

        ei = expected_improvement(mu, std, np.max(y))
        ucb = rf_mu + 2.5 * std

        score = 0.7 * ei + 0.3 * ucb

        return candidates[np.argmax(score)], X[np.argmax(y)], "AdaptiveMix"


# ================================================================
# DIRICHLET / SIMPLEX / ANISOTROPIC / TURBO_RF / LOCAL_GLOBAL
# ================================================================

class DirichletStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed):

        candidates = dirichlet_sampling(params.get("n_cand", 5000), dim, seed)
        mu, std = gp.predict(candidates, return_std=True)

        ei = expected_improvement(mu, std, np.max(y))
        return candidates[np.argmax(ei)], X[np.argmax(y)], "Dirichlet"


class SimplexAdaptiveStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed):

        best = X[np.argmax(y)]
        candidates = simplex_sampling(best, params.get("n_cand", 5000), dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "Simplex"


class AnisotropicStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed):

        best = X[np.argmax(y)]
        candidates = anisotropic_sampling(best, X, params.get("n_cand", 8000), dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "Anisotropic"


class TurboRFStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed):

        rf = RandomForestRegressor(n_estimators=500, random_state=seed)
        rf.fit(X, y)

        center = X[np.argmax(y)]
        r = params.get("tr_radius", 0.1)

        low = np.maximum(0, center - r)
        high = np.minimum(1, center + r)

        candidates = np.random.uniform(low, high, (params.get("n_local", 5000), dim))

        mu, std = rf_mean_std(rf, candidates)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], center, "TURBO_RF_EI"

#########
class LocalGlobalStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed):

        best = X[np.argmax(y)]
        n_total = params.get("n_cand", 20000)

        n_local = int(n_total * params.get("prop_local", 0.7))
        n_global = n_total - n_local

        radius = params.get("radius", 0.1)

        local = np.clip(
            best + np.random.uniform(-radius, radius, (n_local, dim)),
            0, 1
        )

        global_c = np.random.uniform(0, 1, (n_global, dim))

        candidates = np.vstack([local, global_c])

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "LOCAL_GLOBAL_GP"