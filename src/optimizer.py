# ================================================================
#DATA-DRIVEN BAYESIAN OPTIMIZATION ENGINE
# ================================================================

import os
import numpy as np

from src.config import Config
from src.strategy import StrategySelector

from src.utils.sampling import set_seed
from src.utils.visualization import generate_plots, plot_global_best
from src.utils.io_utils import (save_weekly_report, save_points_txt, log_strategy, save_best_point)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# ================================================================
# DATA LOADER
# ================================================================

def load_function_data(function_name, week):
    """
    Load cumulative data for a given function and week.
    """

    path = os.path.join(
        Config.COMBINED_DATA,
        f"week_{week}"
    )

    fname = function_name

    X_path = os.path.join(path, f"{fname}_combined_inputs.npy")
    y_path = os.path.join(path, f"{fname}_combined_outputs.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing data for {fname} in week {week}")

    X = np.load(X_path)
    y = np.load(y_path)

    return X, y


# ================================================================
# OPTIMIZER ENGINE (DATA-DRIVEN)
# ================================================================

class OptimizerEngine:

    def __init__(self, function_name):

        self.function_name = function_name
        self.week = Config.CURRENT_WEEK
        self.seed = Config.RANDOM_SEED

        self.X = None
        self.y = None
        self.gp = None

        set_seed(self.seed)

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    def load_data(self):

        self.X, self.y = load_function_data(self.function_name, self.week)

        print(f"[DATA] {self.function_name}: {len(self.y)} samples loaded")

    # ------------------------------------------------------------
    # FIT MODEL
    # ------------------------------------------------------------
    def fit_gp(self):

        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=1.5),
            alpha=1e-3,
            normalize_y=True,
            random_state=self.seed
        )

        self.gp.fit(self.X, self.y)

    # ------------------------------------------------------------
    # STRATEGY CONTROL
    # ------------------------------------------------------------
    def requires_gp(self, strategy_name):

        no_gp = [
            "RF_EI",
            "TURBO_RF_EI",
            "ADAPTIVE_MIX"
        ]

        return strategy_name not in no_gp

    # ------------------------------------------------------------
    # MAIN EXECUTION
    # ------------------------------------------------------------
    def run(self):

        print(f"\n[START] {self.function_name} | Week {self.week}")

        # Load historical data
        self.load_data()

        # Get strategy config
        config = Config.get_strategy_params(self.function_name, self.week)
        strategy_name = config["strategy"]
        params = config.get("params", {})

        if self.requires_gp(strategy_name):
            self.fit_gp()
        else:
            self.gp = None

        # --------------------------------------------------------
        # SINGLE STEP BO (suggest next point)
        # --------------------------------------------------------
        x_next, center, label = StrategySelector.execute(
            strategy_name=strategy_name,
            gp=self.gp,
            X=self.X,
            y=self.y,
            params=params,
            dim=self.X.shape[1],
            seed=self.seed
        )

        log_strategy(self.function_name, self.week, label)

        print(f"[RESULT] Suggested next point for {self.function_name}")

        return self.X, self.y, x_next, label

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------
    def save(self, X, y, x_next, strategy_name):

        function = self.function_name
        week = self.week

        best_idx = np.argmax(y)
        best_point = X[best_idx]
        best_value = np.max(y)

        # --------------------------------------------------------
        # Visualization
        # --------------------------------------------------------
        generate_plots(function, week, X, y)

        # --------------------------------------------------------
        # Reporting
        # --------------------------------------------------------
        save_weekly_report(
            function=function,
            week=week,
            strategy=strategy_name,
            gp=self.gp,
            X=X,
            y=y,
            best_point=x_next  # next point is what matters
        )

        # --------------------------------------------------------
        # Simulator input (CRITICAL)
        # --------------------------------------------------------
        save_points_txt(function, x_next, week)

        # --------------------------------------------------------
        # Best point tracking
        # --------------------------------------------------------
        save_best_point(function, best_point, best_value, week)

        # --------------------------------------------------------
        # Global evolution
        # --------------------------------------------------------
        plot_global_best(function)


# ================================================================
# MULTI-FUNCTION RUNNER
# ================================================================

def run_all():
    """
    Run optimization for all functions defined in Config.
    """

    functions = Config.get_functions_to_run()

    print("\n[RUNNING ALL FUNCTIONS]")
    print(f"Functions: {functions}")

    for fname in functions:

        try:
            engine = OptimizerEngine(fname)

            X, y, x_next, strategy = engine.run()

            engine.save(X, y, x_next, strategy)

        except Exception as e:
            print(f"[ERROR] {fname} failed: {e}")