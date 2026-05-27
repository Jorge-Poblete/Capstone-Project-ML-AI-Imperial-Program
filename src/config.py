# ================================================================
# CONFIGURATION MODULE
# ================================================================

import os
class Config:

    # ============================================================
    # MAIN
    # ============================================================

    CURRENT_WEEK = 1
    FUNCTIONS_TO_RUN = "all"

    MAIN_PATH = r'C:\Users\JOPOB\JOPOB\15 Imperial_Program\01 Capstone_Project_Imperial_Program'

    BASE_PATH = os.path.join(MAIN_PATH, "data")
    COMBINED_DATA = os.path.join(BASE_PATH, "combined_data")

    OUTPUT_BASE = os.path.join(MAIN_PATH, "outputs")

    INPUTS_BASE = os.path.join(OUTPUT_BASE, "input_for_simulator")
    PLOTS_BASE = os.path.join(OUTPUT_BASE, "plots")
    REPORTS_BASE = os.path.join(OUTPUT_BASE, "reports")

    BEST_POINT_BASE = os.path.join(PLOTS_BASE, "best_point")

    # ============================================================
    # GLOBAL
    # ============================================================

    RANDOM_SEED = 20260401

    # ============================================================
    # STRATEGIES
    # ============================================================

    FUNCTION_STRATEGIES = {

        "function_1": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 10000, "sampling": "random", "top_k": 3, "adaptive_af": True}},
            2: {"strategy": "EI_LOCAL", "params": {"r_local": 0.08}},
            3: {"strategy": "MULTI_LEVEL", "params": {"global": 5000, "meso": 4000, "local": 3000}},
            4: {"strategy": "MULTI_LEVEL", "params": {"global": 5000, "meso": 4000, "local": 3000}},
            5: {"strategy": "REFINEMENT", "params": {"r_local": 0.08}},
            6: {"strategy": "DOUBLE_RADIUS", "params": {"r_small": 0.08, "r_big": 0.16}},
            7: {"strategy": "TURBO", "params": {"tr_init": 0.08}},
            8: {"strategy": "TURBO", "params": {"tr_init": 0.08, "fixed_tr": True}},
            9: {"strategy": "REFINEMENT", "params": {"r_local": 0.05}},
        },

        "function_2": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 10000}},
            2: {"strategy": "MULTI_LEVEL", "params": {"global": 8000, "meso": 2000, "local": 2000}},
            3: {"strategy": "MULTI_LEVEL", "params": {"global": 8000, "meso": 2000, "local": 2000}},
            4: {"strategy": "MULTI_LEVEL", "params": {"global": 8000, "meso": 2000, "local": 2000}},
            5: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 15000}},
            6: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 20000}},
            7: {"strategy": "RF_EI", "params": {"global": 20000}},
            8: {"strategy": "RF_EI", "params": {"global": 25000}},
            9: {"strategy": "REFINEMENT", "params": {"r_local": 0.05}},
        },

        "function_3": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 15000}},
            2: {"strategy": "MULTI_LEVEL", "params": {"global": 12000, "meso": 4000}},
            3: {"strategy": "MULTI_LEVEL", "params": {"global": 12000, "meso": 6000}},
            4: {"strategy": "MULTI_LEVEL", "params": {"global": 12000, "meso": 6000}},
            5: {"strategy": "EI_LOCAL", "params": {"r_local": 0.12}},
            6: {"strategy": "DOUBLE_RADIUS", "params": {"r_small": 0.08, "r_big": 0.20}},
            7: {"strategy": "TURBO", "params": {"tr_init": 0.12}},
            8: {"strategy": "TURBO", "params": {"tr_init": 0.08}},
            9: {"strategy": "RF_EI", "params": {"global": 25000}},
        },

        "function_4": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 20000}},
            2: {"strategy": "RF_EI", "params": {"global": 30000}},
            3: {"strategy": "TPE", "params": {"n_cand": 30000}},
            4: {"strategy": "TPE", "params": {"n_cand": 30000}},
            5: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 25000}},
            6: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 25000}},
            7: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.20}},
            8: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.15}},
            9: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.12}},
        },

        "function_5": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 15000}},
            2: {"strategy": "EI_LOCAL", "params": {"r_local": 0.10}},
            3: {"strategy": "MULTI_LEVEL", "params": {"global": 10000}},
            4: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 20000}},
            5: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 20000}},
            6: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 20000}},
            7: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.15}},
            8: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.08}},
            9: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.04}},
        },

        "function_6": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 20000}},
            2: {"strategy": "DIRICHLET", "params": {"n_cand": 20000}},
            3: {"strategy": "DIRICHLET", "params": {"n_cand": 20000}},
            4: {"strategy": "SIMPLEX", "params": {"n_cand": 15000}},
            5: {"strategy": "SIMPLEX", "params": {"n_cand": 10000}},
            6: {"strategy": "DIRICHLET", "params": {"n_cand": 10000}},
            7: {"strategy": "ANISOTROPIC", "params": {"n_cand": 20000}},
            8: {"strategy": "ANISOTROPIC", "params": {"n_cand": 25000}},
            9: {"strategy": "ANISOTROPIC", "params": {"n_cand": 30000}},
        },

        "function_7": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 30000}},
            2: {"strategy": "TPE", "params": {"n_cand": 40000}},
            3: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 60000}},
            4: {"strategy": "ADAPTIVE_MIX", "params": {"n_cand": 60000}},
            5: {"strategy": "TPE", "params": {"n_cand": 40000}},
            6: {"strategy": "INITIAL", "params": {"n_cand": 30000}},
            7: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.25, "n_local": 6000}},
            8: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.15, "n_local": 8000}},
            9: {"strategy": "TURBO_RF_EI", "params": {"tr_radius": 0.09, "n_local": 13000}},
        },

        "function_8": {
            1: {"strategy": "INITIAL", "params": {"n_cand": 40000}},
            2: {"strategy": "TPE", "params": {"n_cand": 60000}},
            3: {"strategy": "MULTI_LEVEL", "params": {"global": 30000}},
            4: {"strategy": "MULTI_LEVEL", "params": {"global": 30000}},
            5: {"strategy": "TPE", "params": {"n_cand": 60000}},
            6: {"strategy": "INITIAL", "params": {"n_cand": 40000}},
            7: {"strategy": "INITIAL", "params": {"n_cand": 40000}},
            8: {"strategy": "LOCAL_GLOBAL_GP", "params": {"prop_local": 0.7, "radius": 0.10}},
            9: {"strategy": "LOCAL_GLOBAL_GP", "params": {"prop_local": 0.9, "radius": 0.03}},
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