# ================================================================
# CONFIGURATION MODULE
# ================================================================

import os
class Config:

    CURRENT_WEEK = 13
    FUNCTIONS_TO_RUN = "all"

    # ROOT PROJECT PATH (critical fix)
    MAIN_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # ============================================================
    # DATA PATHS
    # ============================================================

    BASE_PATH = os.path.join(MAIN_PATH, "data")
    COMBINED_DATA = os.path.join(BASE_PATH, "combined_data")

    OUTPUT_BASE = os.path.join(MAIN_PATH, "outputs")

    PLOTS_BASE = os.path.join(OUTPUT_BASE, "plots")
    REPORTS_BASE = os.path.join(OUTPUT_BASE, "reports")

    # ============================================================
    # GLOBAL SETTINGS
    # ============================================================

    RANDOM_SEED = 20260401

    # ============================================================
    # STRATEGIES
    # ============================================================

    FUNCTION_STRATEGIES = {

        "function_1": {
            1: {"strategy": "GLOBAL_EXPLORATION", "params": {"n_cand": 10000, "sampling": "random", "af": "EI/UCB"}},# Initial global exploration → no signal detected (outputs ~0)
            2: {"strategy": "EI_LOCAL", "params": {"r_local": 0.08, "xi": 0.01, "high_density": True}}, # Introduction of local EI → identifies promising region
            3: {"strategy": "EVALUATION", "params": {"source": "week_2_best"}}, # Evaluate best candidate from Week 2 → global optimum discovered
            4: {"strategy": "GLOBAL_EXPLORATION", "params": {"n_cand": 8000, "af": "UCB"}}, # Return to global exploration → loses focus around optimum
            5: {"strategy": "WEAK_LOCAL_SEARCH", "params": {"r_local": 0.08}},  # Attempted local refinement → does not replicate successful setup
            6: {"strategy": "DOUBLE_RADIUS", "params": {"r_small": 0.08, "r_big": 0.16}}, # Over-exploration → large radius moves away from optimum
            7: {"strategy": "TURBO", "params": {"tr_init": 0.08, "adaptive": True}}, # Methodological correction → trust-region improves focus
            8: {"strategy": "TURBO_LOCAL", "params": {"r_local": 0.08, "fixed_tr": True}},  # Reintroduce local EI within TR → stable convergence behavior
            9: {"strategy": "LOCAL_REFINEMENT", "params": {"r_local": 0.05}},  # Local refinement → no further improvement (convergence begins)
            10: {"strategy": "LOCAL_REFINEMENT", "params": {"r_local": 0.05}}, # Continued refinement → stable, no improvement
            11: {"strategy": "FINE_LOCAL_SEARCH", "params": {"r_local": 0.02}}, # Fine-grained search → sharp drop confirms narrow peak
            12: {"strategy": "LOCAL_MULTI_POINT", "params": {"r_local": 0.02, "n_points": 4, "diversity": True}},# Multi-point validation → all nearby points perform worse
            13: {"strategy": "FINAL_VALIDATION","params": {"method": "manual_perturbation", "delta": 0.003, "n_points": 4}}, # Final validation → controlled perturbations confirm optimality

        },

        "function_2": {

            1: {"strategy": "INITIAL_GP", "params": {"model": "GP", "af": "EI_UCB", "n_cand": 10000, "phase": "explore"}},#Initial Gaussian Process model with global exploration # Objective: establish baseline performance and explore the full search space
            2: {"strategy": "GP_GLOBAL", "params": {"model": "GP", "af": "EI_UCB", "n_cand": 10000, "phase": "explore"}},# Continued global exploration with GP  # No improvement observed → early indication of model mismatch
            3: {"strategy": "GP_GLOBAL_TR","params": {"model": "GP", "af": "EI_UCB", "global": 8000, "local": 2000, "phase": "adapt"}},#Introduced Trust Region (TR) to balance local refinement  # Attempt to focus search around promising regions
            4: {"strategy": "GP_TR", "params": {"model": "GP", "af": "EI_UCB", "global": 7000, "local": 3000, "phase": "adapt"}},#Increased local sampling via TR   # Goal: improve exploitation, but still limited by GP assumptions
            5: {"strategy": "GP_TR_EXPLOIT", "params": {"model": "GP", "af": "EI_UCB", "global": 6000, "local": 4000, "phase": "adapt"}},#More exploitation (local search dominates)   # Still no breakthrough → GP continues to struggle with irregular surface
            6: {"strategy": "GP_TR_EXPAND","params": {"model": "GP", "af": "EI_UCB", "global": 10000, "local": 5000, "phase": "adapt"}},# Expanded exploration again due to stagnation  # Attempt to escape local regions and rediscover structure
            7: {"strategy": "RF_EI_GLOBAL", "params": {"model": "RF", "af": "EI", "n_cand": 15000, "phase": "transition"}},# Critical transition → switch from GP to Random Forest  # Motivation: function shows non-smooth, irregular behavior
            8: {"strategy": "RF_EI_GLOBAL","params": {"model": "RF", "af": "EI", "n_cand": 20000, "phase": "explore"}},# Increased global exploration with RF surrogate  # Aim: leverage robustness of RF to better scan space
            9: {"strategy": "RF_EI_DISCOVERY","params": {"model": "RF", "af": "EI", "global": 20000, "phase": "discover"}},# Breakthrough → new global optimum discovered (~0.633)  # Confirms effectiveness of RF + broader exploration
            10: {"strategy": "RF_EI_LOCAL", "params": {"model": "RF", "af": "EI", "global": 12000, "local": 8000, "tr_radius": 0.12, "phase": "refine"}},#Begin local exploitation around new optimum   # However, slight overshooting observed in high x2 region
            11: {"strategy": "RF_EI_LOCAL_FINE", "params": {"model": "RF", "af": "EI", "global": 10000, "local": 10000, "tr_radius": 0.08,"phase": "refine"}},# Increased local density for fine-grained exploration  # Revealed strong sensitivity → sharp peak / narrow optimum
            12: {"strategy": "RF_EI_BALANCED","params": {"model": "RF", "af": "EI", "global": 14000, "local": 10000, "tr_radius": 0.05,"phase": "refine"}},# Rebalanced exploration after local overshoot # Confirmed that optimal region is asymmetric and highly localized
            13: {"strategy": "RF_EI_FINAL","params": {"model": "RF", "af": "EI", "global": 6000, "local": 14000, "tr_radius": 0.03,"phase": "converge"}},# Final convergence phase.Heavy local exploitation with very small radius → confirm optimum and close project
        },

        "function_3": {
            1: {"strategy": "INITIAL_EI", "params": {"n_cand": 15000}},  # Initial BO with EI, early best found
            2: {"strategy": "EARLY_EXPLOIT", "params": {"global": 12000}},   # Premature exploitation around initial optimum
            3: {"strategy": "EI_NOISY", "params": {"global": 12000, "noise": 0.01}},  # Added noise to force exploration
            4: {"strategy": "EI_STD_MIX", "params": {"global": 12000, "std_weight": 0.3}},  # Mixed EI + uncertainty
            5: {"strategy": "EI_LOCAL", "params": {"r_local": 0.12}},  # Local search around best (GP still active)
            6: {"strategy": "DUAL_TR", "params": {"r_small": 0.08, "r_big": 0.20}},  # Two-scale local exploration
            7: {"strategy": "TS_TURBO", "params": {"tr_init": 0.12}},  # Switch to Thompson Sampling + TuRBO Lite
            8: {"strategy": "TS_TURBO", "params": {"tr_init": 0.08}},  # Stronger local exploitation with smaller TR
            9: {"strategy": "RF_EI_GLOBAL", "params": {"global": 30000}},  # Switch to Random Forest surrogate + global EI
            10: {"strategy": "RF_EI_EXPLORATION", "params": {"global": 50000, "xi": 0.01}},  # Over-exploration phase
            11: {"strategy": "RF_EI_BALANCED", "params": {"global": 20000, "xi": 0.001}},  # Reduced exploration, partial focus
            12: {"strategy": "RF_EI_FOCUSED", "params": {"global": 20000, "local_radius": 0.10}},  # Identification of promising region
            13: {"strategy": "RF_EI_FINAL_LOCAL", "params": {"local_radius": 0.05, "batch": 2, "xi": 0.0001}},  # Final local refinement (exploit only)
        },

        "function_4": {
            1: {"strategy": "INITIAL_GLOBAL", "params": {"n_cand": 20000}},  # Initial global exploration (no model structure)
            2: {"strategy": "RF_UCB", "params": {"n_cand": 30000, "kappa": 2.5}},  # First improvement with RF + UCB (robust to noise)
            3: {"strategy": "TPE_GLOBAL", "params": {"n_cand": 30000}},  # Attempt TPE (degenerates, no learning signal),failed density modeling
            4: {"strategy": "TPE_GLOBAL", "params": {"n_cand": 30000}},  # TPE continues but still degenerate
            5: {"strategy": "RF_UCB", "params": {"n_cand": 30000, "kappa": 2.5}},  #Back to RF + UCB (stable but no new optima), plateau starts
            6: {"strategy": "RF_UCB", "params": {"n_cand": 30000, "kappa": 2.5}},  # Same RF + UCB (over-exploration effect begins), variance-driven sampling
            7: {"strategy": "RF_UCB", "params": {"n_cand": 30000, "kappa": 2.5}},  # Transition phase (still global mindset, unstable), poor exploration
            8: {"strategy": "TURBO_TS", "params": {"tr_radius": 0.20}},  # TuRBO introduced (trust-region, local BO begins)
            9: {"strategy": "TURBO_TS", "params": {"tr_radius": 0.20}},  # Optimal region found (best performance)
            10: {"strategy": "TURBO_TS", "params": {"tr_radius": 0.20}},  #Same TR → too large for refinement (first deviation)
            11: {"strategy": "TURBO_TS", "params": {"tr_radius": 0.15}},  # Still large TR → continued oscillation
            12: {"strategy": "TURBO_TS", "params": {"tr_radius": 0.15}},  #Over-exploration persists → degradation
            13: {"strategy": "TURBO_TS", "params": {"tr_radius": 0.06}},  # Final refinement (ultra local exploitation)
        },

        "function_5": {


            1: {"strategy": "INITIAL_GLOBAL", "params": {"n_cand": 15000}}, # Week 1: initial global exploration (random / uniform)
            2: {"strategy": "EI_LOCAL", "params": {"r_local": 0.10}},# Week 2: exploit direction found using EI + local trust region
            3: {"strategy": "GLOBAL_LOCAL", "params": {"n_global": 10000, "r_local": 0.20}}, # Week 3: global + local refinement (UCB / exploration)
            4: {"strategy": "MIXED_AF", "params": {"alpha": 0.6, "kappa": 1.8}}, # Week 4: best configuration → mixed AF (UCB + EI)
            5: {"strategy": "EXPLOIT", "params": {"r_local": 0.15}},# Week 5: pure exploitation around optimum
            6: {"strategy": "UNCERTAINTY_OVERRIDE", "params": {"mode": "argmax_std"}},# Week 6: EI collapse → uncertainty-based override (failed)
            7: {"strategy": "TURBO_1", "params": {"tr_radius": 0.10}},# Week 7: recovery using TuRBO‑1 (dynamic trust region)
            8: {"strategy": "TURBO_LOCK", "params": {"tr_radius": 0.05, "ei_lock": 1e-6}}, # Week 8: TuRBO‑1 + exploitation lock
            9: {"strategy": "CONVERGENCE", "params": {"tr_radius": 0.03, "ei_lock": 1e-7}},# Week 9: strong convergence phase (reduced TR)
            10: {"strategy": "LOCAL_VALIDATION", "params": {"tr_radius": 0.02, "n_cand": 100}},# Week 10: local validation (micro trust region)
            11: {"strategy": "FINAL_CONFIRMATION", "params": {"tr_radius": 0.01, "ei_lock": 1e-9}}, # Week 11: convergence confirmation (near-zero EI)
            12: {"strategy": "STABILITY_CHECK", "params": {"tr_radius": 0.008, "n_cand": 50}}, # Week 12: stability check (plateau validation)
            13: {"strategy": "TERMINATION", "params": {"stop": True}}   # Week 13: formal termination (stop optimization)
        },

        "function_6": {
            1: {"strategy": "GLOBAL_EI", "params": {"n_cand": 50000, "kernel": "RBF", "xi": 0.01}},  # Optimal found (EI_Balance)
            2: {"strategy": "DIRICHLET", "params": {"n_cand": 30000, "kernel": "Matern_ARD", "xi": 0.01}}, # Increased complexity (signal loss)
            3: {"strategy": "DIRICHLET", "params": {"n_cand": 30000, "kernel": "Matern_ARD", "xi": 0.01}}, # Structured exploration (drift)
            4: {"strategy": "WINDOW", "params": {"n_cand": 20000, "kernel": "Matern", "xi": 0.01}},  # Windowed search
            5: {"strategy": "SIMPLEX", "params": {"n_cand": 20000, "kernel": "Matern", "xi": 0.01}},# Restricted exploration
            6: {"strategy": "GLOBAL_EI", "params": {"n_cand": 30000, "kernel": "RBF", "xi": 0.01}},  # Return to base model
            7: {"strategy": "TRUST_REGION", "params": {"n_cand": 30000, "kernel": "RBF", "radius": 0.05}},  # Local TR (wrong region)
            8: {"strategy": "ANISOTROPIC", "params": {"n_cand": 30000, "scale": [0.04, 0.04, 0.03, 0.01, 0.02]}}, # Initial local refinement
            9: {"strategy": "ANISOTROPIC_EI", "params": {"n_cand": 30000, "score": "0.7EI+0.3mean"}}, # EI balance local attempt
            10: {"strategy": "FILTERED_LOCAL", "params": {"n_cand": 32768, "filter": 30}},  # Dataset filtering
            11: {"strategy": "RECENTER_LOCAL", "params": {"n_cand": 32768, "scale": [0.015, 0.015, 0.01, 0.002, 0.005]}},  # Re-centering attempt
            12: {"strategy": "LOCAL_REFINEMENT", "params": {"n_cand": 32768, "score": "EI+mean+prox"}}, # Stable suboptimal region
            13: {"strategy": "VALIDATION", "params": {"center": "EI_balance", "perturb": 0.01, "n_points": 3}} # Final validation
        },

        "function_7": {

            1: {"strategy": "GP_EI_GLOBAL", "params": {"n_cand": 30000}},  # broad exploration, high uncertainty
            2: {"strategy": "GP_EI_GLOBAL", "params": {"n_cand": 40000}},  # optimal region discovered
            3: {"strategy": "RF_EI_UCB", "params": {"n_cand": 60000}},  # reduced exploration
            4: {"strategy": "RF_EI_UCB", "params": {"n_cand": 60000}},  # plateau continues
            5: {"strategy": "RF_EI_UCB", "params": {"n_cand": 40000}},  # no new regions explored
            6: {"strategy": "GP_EI_GLOBAL", "params": {"n_cand": 30000}},  # exploration resumes but inefficient
            7: {"strategy": "LOCAL_EI", "params": {"n_cand": 30000}},  # no structural change yet
            8: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.25, "n_local": 6000}},  # adaptive local BO
            9: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.15, "n_local": 8000}},  # local refinement
            10: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.09, "n_local": 13000}},  # drift inside region
            11: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.06, "n_local": 18000}},  # recovery near optimum
            12: {"strategy": "TURBO_RF_EI_REFINED", "params": {"tr_radius": 0.04, "n_local": 25000}}, # almost converged
            13: {"strategy": "TURBO_FINAL_LOCK", "params": {"tr_radius": 0.02, "n_local": 35000}} # deterministic refinement
        },

        "function_8": {
            1: {"strategy": "INITIAL_GP_EI", "params": {"n_cand": 40000, "xi": 0.01}}, # Week 1: First GP+EI → major improvement
            2: {"strategy": "TPE", "params": {"n_cand": 60000}}, # Week 2: Guided exploration → marginal improvement
            3: {"strategy": "MULTI_LEVEL", "params": {"n_cand": 80000}}, # Week 3: Over-exploration → no gains
            4: {"strategy": "MULTI_LEVEL", "params": {"n_cand": 80000}}, # Week 4: Continued exploration → stagnation
            5: {"strategy": "TPE", "params": {"n_cand": 60000}},  # Week 5: TPE reuse → no improvement
            6: {"strategy": "INITIAL_GP_EI", "params": {"n_cand": 40000, "xi": 0.01}}, # Week 6: Reset with GP
            7: {"strategy": "INITIAL_GP_EI", "params": {"n_cand": 40000, "xi": 0.01}}, # Week 7: Breakthrough → new region
            8: {"strategy": "LOCAL_GLOBAL_GP", "params": {"prop_local": 0.7, "radius": 0.10, "xi": 0.01}}, # Week 8: Start refinement
            9: {"strategy": "LOCAL_GLOBAL_GP","params": {"prop_local": 0.9, "radius": 0.03, "xi": 0.005}}, # Week 9: Strong local improvement
            10: {"strategy": "LOCAL_GLOBAL_GP","params": {"prop_local": 0.9, "radius": 0.03, "xi": 0.002}},  # Week 10: Near optimum
            11: {"strategy": "ULTRA_LOCAL_GP","params": {"prop_local": 0.95, "radius": 0.015, "xi": 0.001, "elite": 0.003}}, # Week 11: Ultra-local → overshoot
            12: {"strategy": "ULTRA_LOCAL_GP","params": {"prop_local": 0.97, "radius": 0.008, "xi": 0.0005, "elite": 0.0015}}, # Week 12: New optimum ✅
            13: {"strategy": "FINAL_EXPLOIT","params": {"prop_local": 0.98, "radius": 0.005, "xi": 0.0002, "elite": 0.0008}},# Week 13: Final convergence
        }
    }
    # ============================================================

    # HELPERS
    # ============================================================

    @classmethod
    def get_functions_to_run(cls):

        if cls.FUNCTIONS_TO_RUN == "all":
            return [f"function_{i}" for i in range(1, 9)]

        if isinstance(cls.FUNCTIONS_TO_RUN, str):
            return [cls.FUNCTIONS_TO_RUN]

        return cls.FUNCTIONS_TO_RUN


    @classmethod
    def get_strategy_params(cls, function_name, week=None):

        week = week or cls.CURRENT_WEEK

        function_config = cls.FUNCTION_STRATEGIES[function_name]

        if week not in function_config:
            available_weeks = sorted(function_config.keys())
            week = max([w for w in available_weeks if w <= week])

        return function_config[week]
