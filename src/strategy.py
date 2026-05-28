
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
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
    def execute(strategy_name, gp, X, y, params, dim, seed, function_name=None):

        mapping = {
            # Base strategies (implemented)
            "INITIAL": InitialStrategy,
            "GLOBAL_EXPLORATION": InitialStrategy,
            "INITIAL_GLOBAL": InitialStrategy,
            "INITIAL_GP": InitialStrategy,
            "INITIAL_EI": InitialStrategy,
            "GLOBAL_EI": InitialStrategy,
            "INITIAL_GP_EI": InitialStrategy,

            "EI_LOCAL": EILocalStrategy,
            "LOCAL_EI": EILocalStrategy,
            "LOCAL_REFINEMENT": EILocalStrategy,
            "WEAK_LOCAL_SEARCH": EILocalStrategy,
            "FINE_LOCAL_SEARCH": EILocalStrategy,

            "MULTI_LEVEL": MultiLevelStrategy,

            "DOUBLE_RADIUS": DoubleRadiusStrategy,

            "TURBO": TurboStrategy,
            "TURBO_TS": TurboStrategy,
            "TURBO_LOCAL": TurboStrategy,
            "TURBO_1": TurboStrategy,
            "TURBO_LOCK": TurboStrategy,
            "CONVERGENCE": TurboStrategy,

            "RF_EI": RandomForestStrategy,
            "RF_EI_GLOBAL": RandomForestStrategy,
            "RF_EI_LOCAL": RandomForestStrategy,
            "RF_EI_BALANCED": RandomForestStrategy,
            "RF_EI_FINAL": RandomForestStrategy,
            "RF_EI_DISCOVERY": RandomForestStrategy,

            "TPE": TPEStrategy,

            "DIRICHLET": DirichletStrategy,
            "SIMPLEX": SimplexAdaptiveStrategy,
            "ANISOTROPIC": AnisotropicStrategy,

            "TURBO_RF_EI": TurboRFStrategy,
            "TURBO_RF_EI_REFINED": TurboRFStrategy,
            "TURBO_FINAL_LOCK": TurboRFStrategy,

            "LOCAL_GLOBAL_GP": LocalGlobalStrategy,
        }

        if strategy_name not in mapping:
            print(f"[WARNING] Strategy '{strategy_name}' not implemented → fallback to EI_LOCAL")

        strategy = mapping.get(strategy_name, EILocalStrategy)

        return strategy.run(gp, X, y, params, dim, seed, function_name)

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
# INITIAL / GLOBAL
# ================================================================
class InitialStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        n = params.get("n_cand", 10000)

        if params.get("sampling") == "random":
            candidates = np.random.uniform(0, 1, (n, dim))
        else:
            candidates = sobol_sampling(n, dim, seed)

        return candidates[0], None, "INITIAL"

# ================================================================
# EI LOCAL
# ================================================================
class EILocalStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        center = X[np.argmax(y)]
        radius = params.get("r_local", params.get("local_radius", 0.1))

        candidates = polar_sampling(center, radius, 5000, dim, seed)

        if gp is None:
            return candidates[0], center, "LOCAL"

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y), xi=params.get("xi", 0.01))

        return candidates[np.argmax(ei)], center, "EI_LOCAL"

# ================================================================
# MULTI LEVEL
# ================================================================
class MultiLevelStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        best = X[np.argmax(y)]

        global_c = sobol_sampling(params.get("n_cand", 10000), dim, seed)
        local_c = polar_sampling(best, 0.1, 5000, dim, seed)

        candidates = np.vstack([global_c, local_c])

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "MULTI_LEVEL"

# ================================================================
# DOUBLE RADIUS
# ================================================================
class DoubleRadiusStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        best = X[np.argmax(y)]

        small = polar_sampling(best, params.get("r_small", 0.05), 3000, dim, seed)
        big = polar_sampling(best, params.get("r_big", 0.2), 3000, dim, seed)

        candidates = np.vstack([small, big])

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "DOUBLE_RADIUS"

# ================================================================
# TURBO (SIMPLE)
# ================================================================
class TurboStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        center = X[np.argmax(y)]
        r = params.get("tr_init", params.get("radius", 0.1))

        low = np.maximum(0, center - r)
        high = np.minimum(1, center + r)

        candidates = np.random.uniform(low, high, (5000, dim))

        if gp is None:
            return candidates[0], center, "TURBO"

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], center, "TURBO"

# ================================================================
# RF EI
# ================================================================
class RandomForestStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        rf = RandomForestRegressor(n_estimators=300, random_state=seed)
        rf.fit(X, y)

        candidates = sobol_sampling(params.get("n_cand", 20000), dim, seed)

        mu, std = rf_mean_std(rf, candidates)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], X[np.argmax(y)], "RF_EI"

# ================================================================
# TPE
# ================================================================
class TPEStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        candidates = sobol_sampling(params.get("n_cand", 20000), dim, seed)

        if len(y) < 10:
            return candidates[0], X[np.argmax(y)], "TPE"

        threshold = np.quantile(y, 0.8)

        good = X[y >= threshold]
        bad = X[y < threshold]

        if len(good) < 5 or len(bad) < 5:
            return candidates[0], X[np.argmax(y)], "TPE"

        gmm_good = GaussianMixture(2).fit(good)
        gmm_bad = GaussianMixture(2).fit(bad)

        score = np.exp(gmm_good.score_samples(candidates)) / (
            np.exp(gmm_bad.score_samples(candidates)) + 1e-12
        )

        return candidates[np.argmax(score)], X[np.argmax(y)], "TPE"

# ================================================================
# DIRICHLET / SIMPLEX / ANISOTROPIC
# ================================================================
class DirichletStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        candidates = dirichlet_sampling(params.get("n_cand", 5000), dim, seed)
        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], X[np.argmax(y)], "Dirichlet"

class SimplexAdaptiveStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        best = X[np.argmax(y)]
        candidates = simplex_sampling(best, params.get("n_cand", 5000), dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "Simplex"

class AnisotropicStrategy:
    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        best = X[np.argmax(y)]
        candidates = anisotropic_sampling(best, X, params.get("n_cand", 8000), dim, seed)

        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y))

        return candidates[np.argmax(ei)], best, "Anisotropic"


# ================================================================
# TURBO RF
# ================================================================
class TurboRFStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        rf = RandomForestRegressor(n_estimators=500, random_state=seed)
        rf.fit(X, y)

        center = X[np.argmax(y)]
        r = params.get("tr_radius", 0.1)

        low = np.maximum(0, center - r)
        high = np.minimum(1, center + r)

        candidates = np.random.uniform(
            low, high,
            (params.get("n_local", 5000), dim)
        )

        mu, std = rf_mean_std(rf, candidates)
        ei = expected_improvement(mu, std, np.max(y), xi=params.get("xi", 0.01))

        return candidates[np.argmax(ei)], center, "TURBO_RF_EI"

# ================================================================
# LOCAL GLOBAL
# ================================================================
class LocalGlobalStrategy:

    @staticmethod
    def run(gp, X, y, params, dim, seed, function_name=None):

        best = X[np.argmax(y)]
        n_total = params.get("n_cand", 20000)

        n_local = int(n_total * params.get("prop_local", 0.7))
        n_global = n_total - n_local

        r = params.get("radius", 0.1)

        local = np.clip(
            best + np.random.uniform(-r, r, (n_local, dim)),
            0, 1
        )
        global_c = np.random.uniform(0, 1, (n_global, dim))
        candidates = np.vstack([local, global_c])
        mu, std = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, std, np.max(y), xi=params.get("xi", 0.01))
        return candidates[np.argmax(ei)], best, "LOCAL_GLOBAL_GP"