
import os
import numpy as np
from src.config import Config

# ================================================================
# BASE PATHS
# ================================================================

OUTPUT_BASE = Config.OUTPUT_BASE
PLOTS_BASE = Config.PLOTS_BASE
REPORTS_BASE = Config.REPORTS_BASE


# ================================================================
# HELPERS
# ================================================================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ================================================================
# FOLDER MANAGEMENT
# ================================================================

def get_plot_folder(function, week):
    return _ensure_dir(os.path.join(PLOTS_BASE, function, f"week_{week}"))


def get_report_folder(function, week):
    return _ensure_dir(os.path.join(REPORTS_BASE, function, f"week_{week}"))


# ================================================================
# WEEKLY REPORT
# ================================================================

def save_weekly_report(function, week, strategy, gp, X, y, best_point):
    folder = get_report_folder(function, week)
    path = os.path.join(
        folder,
        f"{function}_week_{week}_report.txt"
    )
    with open(path, "w") as f:

        f.write("=====================================\n")
        f.write("WEEKLY OPTIMIZATION REPORT\n")
        f.write("=====================================\n\n")

        # --------------------------------------------------------
        # EXPERIMENT
        # --------------------------------------------------------
        f.write("[EXPERIMENT]\n")
        f.write("-------------------------------------\n")
        f.write(f"Function     : {function}\n")
        f.write(f"Week         : {week}\n")
        f.write(f"Strategy     : {strategy}\n")
        f.write(f"Samples      : {len(y)}\n\n")

        # --------------------------------------------------------
        # MODEL
        # --------------------------------------------------------
        f.write("[MODEL]\n")
        f.write("-------------------------------------\n")
        f.write(f"Type         : {type(gp).__name__ if gp else 'RF/Hybrid'}\n")
        f.write(f"Kernel       : {getattr(gp, 'kernel_', 'N/A')}\n\n")

        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------
        f.write("[METRICS]\n")
        f.write("-------------------------------------\n")

        best_val = np.max(y)

        f.write(f"Best Value   : {best_val:.6f}\n")
        f.write(f"Mean         : {np.mean(y):.6f}\n")
        f.write(f"Std          : {np.std(y):.6f}\n\n")

        # --------------------------------------------------------
        # NEXT POINT
        # --------------------------------------------------------
        f.write("[NEXT POINT - SIMULATOR]\n")
        f.write("-------------------------------------\n")
        f.write(" | ".join([f"{v:.6f}" for v in np.ravel(best_point)]) + "\n")

    print(f"[REPORT] {path}")

    return path