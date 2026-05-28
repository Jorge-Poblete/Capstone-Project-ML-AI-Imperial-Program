# ================================================================
# SAMPLING UTILITIES
# ================================================================

import numpy as np
from scipy.stats import qmc

# ================================================================
# GLOBAL SEED
# ================================================================

def set_seed(seed):
    np.random.seed(seed)
# ================================================================
# BASIC SAMPLING
# ================================================================

def sobol_sampling(n_points, dim, seed=42):
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return sampler.random(n_points)

def random_sampling(n_points, dim, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, size=(n_points, dim))

def polar_sampling(center, radius, n_points, dim, seed=42):

    rng = np.random.RandomState(seed)

    directions = rng.normal(size=(n_points, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    directions /= norms

    dist = rng.uniform(0, radius, size=(n_points, 1))

    return np.clip(center + directions * dist, 0, 1)

# ================================================================
# HYBRID SAMPLING
# ================================================================

def local_global_sampling(best, n_total, dim, radius=0.1, prop_local=0.7, seed=42):

    rng = np.random.RandomState(seed)
    n_local = int(n_total * prop_local)
    n_global = n_total - n_local
    local = np.clip(
        best + rng.uniform(-radius, radius, size=(n_local, dim)),
        0, 1
    )
    global_cand = rng.uniform(0, 1, size=(n_global, dim))
    return np.vstack([local, global_cand])


# ================================================================
# ADVANCED SAMPLING
# ================================================================

def dirichlet_sampling(n_points, dim, seed=42, alpha=1.0):
    rng = np.random.RandomState(seed)
    alpha_vec = np.ones(dim) * alpha
    return rng.dirichlet(alpha_vec, size=n_points)

def simplex_sampling(center, n_points, dim, scale=0.1, seed=42):
    rng = np.random.RandomState(seed)
    perturb = rng.dirichlet(np.ones(dim), size=n_points)
    return np.clip(center + scale * (perturb - 1 / dim), 0, 1)

def anisotropic_sampling(center, X, n_points, dim, seed=42):
    rng = np.random.RandomState(seed)
    var = np.var(X, axis=0)
    var = np.maximum(var, 1e-12)
    weights = var / np.sum(var)
    noise = rng.normal(0, weights, size=(n_points, dim))
    return np.clip(center + noise, 0, 1)


# ================================================================
# TURBO SAMPLING
# ================================================================

def trust_region_sampling(center, radius, n_points, dim, seed=42):


    rng = np.random.RandomState(seed)
    low = np.maximum(0, center - radius)
    high = np.minimum(1, center + radius)
    return rng.uniform(low, high, size=(n_points, dim))


# ================================================================
# RANDOM FOREST UTILS
# ================================================================

def rf_mean_std(rf, Xc):
    preds = np.column_stack([
        tree.predict(Xc) for tree in rf.estimators_
    ])
    return preds.mean(axis=1), preds.std(axis=1)