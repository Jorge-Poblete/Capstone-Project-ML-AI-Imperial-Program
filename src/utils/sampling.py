import numpy as np
from scipy.stats import qmc

# -----------------------
# SEED
# -----------------------
def set_seed(seed):
    np.random.seed(seed)

# -----------------------
# BASIC
# -----------------------
def sobol_sampling(n_points, dim, seed=42):
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return sampler.random(n_points)

def polar_sampling(center, radius, n_points, dim, seed=42):
    np.random.seed(seed)
    directions = np.random.normal(size=(n_points, dim))
    directions /= np.linalg.norm(directions, axis=1).reshape(-1, 1)
    dist = np.random.uniform(0, radius, size=(n_points, 1))
    return np.clip(center + directions * dist, 0, 1)

# -----------------------
# ADVANCED
# -----------------------
def local_global_sampling(best, n_total, dim, radius=0.1, prop_local=0.7, seed=42):
    np.random.seed(seed)
    n_local = int(n_total * prop_local)
    n_global = n_total - n_local

    local = np.clip(best + np.random.uniform(-radius, radius, size=(n_local, dim)), 0, 1)
    global_cand = np.random.uniform(0, 1, size=(n_global, dim))

    return np.vstack([local, global_cand])

def dirichlet_sampling(n_points, dim, alpha=1.0, seed=42):
    np.random.seed(seed)
    return np.random.dirichlet(np.ones(dim)*alpha, size=n_points)

def simplex_sampling(center, n_points, dim, scale=0.1, seed=42):
    np.random.seed(seed)
    perturb = np.random.dirichlet(np.ones(dim), size=n_points)
    return np.clip(center + scale*(perturb - 1/dim), 0, 1)

def anisotropic_sampling(center, X, n_points, dim, seed=42):
    np.random.seed(seed)
    var = np.var(X, axis=0)
    w = var / (np.sum(var)+1e-12)
    noise = np.random.normal(0, w, size=(n_points, dim))
    return np.clip(center + noise, 0, 1)

# -----------------------
# RF
# -----------------------
def rf_mean_std(rf, Xc):
    preds = np.column_stack([t.predict(Xc) for t in rf.estimators_])
    return preds.mean(axis=1), preds.std(axis=1)
