
import os
import numpy as np
from src.config import Config
from src.strategy import StrategySelector
from src.utils.sampling import set_seed
from src.utils.visualization import generate_plots
from src.utils.io_utils import save_weekly_report
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ================================================================
# DATA LOADER
# ================================================================

def load_function_data(function_name, week):
    path = os.path.join(Config.COMBINED_DATA, f"week_{week}")
    X_path = os.path.join(path, f"{function_name}_combined_inputs.npy")
    y_path = os.path.join(path, f"{function_name}_combined_outputs.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing data for {function_name} in week {week}")

    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

# ================================================================
# OPTIMIZER ENGINE
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
        print(f"[DATA] {self.function_name} → samples: {len(self.y)}")
    # ------------------------------------------------------------
    # FIT GP
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
            "RF_EI_GLOBAL",
            "RF_EI_LOCAL",
            "RF_EI_BALANCED",
            "RF_EI_FINAL",
            "RF_EI_DISCOVERY",
            "TURBO_RF_EI",
            "TURBO_RF_EI_REFINED",
            "TURBO_FINAL_LOCK",
            "ADAPTIVE_MIX"
        ]

        return strategy_name not in no_gp

    # ------------------------------------------------------------
    # MAIN EXECUTION
    # ------------------------------------------------------------
    def run(self):

        print(f"\n[START] {self.function_name} | Week {self.week}")

        # Load data
        self.load_data()

        # Load strategy config
        config = Config.get_strategy_params(self.function_name, self.week)
        strategy_name = config["strategy"]
        params = config.get("params", {})

        print(f"[STRATEGY] {strategy_name}")

        # Model decision
        if self.requires_gp(strategy_name):
            self.fit_gp()
        else:
            self.gp = None

        # --------------------------------------------------------
        # CORE OPTIMIZATION STEP
        # --------------------------------------------------------
        x_next, center, label = StrategySelector.execute(
            strategy_name=strategy_name,
            gp=self.gp,
            X=self.X,
            y=self.y,
            params=params,
            dim=self.X.shape[1],
            seed=self.seed,
            function_name=self.function_name
        )


        print(f"[RESULT] {self.function_name} → next point generated")

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
        # VISUALIZATION
        # --------------------------------------------------------
        generate_plots(function, week, X, y)

        # --------------------------------------------------------
        # REPORT
        # --------------------------------------------------------
        save_weekly_report(
            function=function,
            week=week,
            strategy=strategy_name,
            gp=self.gp,
            X=X,
            y=y,
            best_point=x_next
        )

# ================================================================
# RUN ALL FUNCTIONS
# ================================================================

def run_all():

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