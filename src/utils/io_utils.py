# ================================================================
# io_utils.py - FULL IO + REPORTING MODULE (TEXT VERSION)
# ================================================================

import os
import numpy as np
from src.config import Config


# ================================================================
# BASE PATHS
# ================================================================

OUTPUT_BASE = Config.OUTPUT_BASE
PLOTS_BASE = Config.PLOTS_BASE
REPORTS_BASE = Config.REPORTS_BASE
INPUTS_BASE = Config.INPUTS_BASE
BEST_POINT_BASE = Config.BEST_POINT_BASE


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ================================================================
# FOLDER MANAGEMENT
# ================================================================

def get_plot_folder(function, week):
    path = os.path.join(PLOTS_BASE, function, f"week_{week}")
    return _ensure_dir(path)


def get_report_folder(function, week):
    path = os.path.join(REPORTS_BASE, function, f"week_{week}")
    return _ensure_dir(path)


def get_input_folder(function, week):
    path = os.path.join(INPUTS_BASE, function, f"week_{week}")
    return _ensure_dir(path)


def get_best_point_folder(function):
    path = os.path.join(BEST_POINT_BASE, function)
    return _ensure_dir(path)


# ================================================================
# SIMULATOR INPUT
# ================================================================

def save_points_txt(function, point, week):

    folder = get_input_folder(function, week)

    path = os.path.join(
        folder,
        f"{function}_week_{week}_next_point.txt"
    )

    point_flat = np.ravel(point)

    with open(path, "w") as f:
        f.write("-".join([f"{v:.6f}" for v in point_flat]))

    print(f"[SIM INPUT] {path}")

    return path


# ================================================================
# BEST POINT (TEXT VERSION)
# ================================================================

def save_best_point(function, best_point, best_value, week):

    folder = get_best_point_folder(function)

    path = os.path.join(folder, f"{function}_best_point.txt")

    with open(path, "w") as f:
        f.write("=====================================\n")
        f.write("BEST POINT SUMMARY\n")
        f.write("=====================================\n\n")

        f.write(f"Function: {function}\n")
        f.write(f"Week: {week}\n")
        f.write(f"Best Value: {best_value:.6f}\n")
        f.write(f"Best Point: {np.ravel(best_point)}\n")

    print(f"[BEST POINT] {path}")


# ================================================================
# STRATEGY LOG
# ================================================================

def log_strategy(function, week, strategy_name):

    folder = os.path.join(REPORTS_BASE, function)
    _ensure_dir(folder)

    path = os.path.join(folder, "strategy_log.txt")

    with open(path, "a") as f:
        f.write(f"Week {week}: {strategy_name}\n")


# ================================================================
# GLOBAL HISTORY (TEXT VERSION)
# ================================================================

def update_global_history(function, week, best_value):

    folder = os.path.join(REPORTS_BASE, function)
    _ensure_dir(folder)

    path = os.path.join(folder, "global_history.txt")

    with open(path, "a") as f:
        f.write(f"Week {week}: Best Value = {best_value:.6f}\n")


# ================================================================
# WEEKLY REPORT (TEXT VERSION)
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

        # ---- Experiment ----
        f.write("EXPERIMENT\n")
        f.write("-------------------------------------\n")
        f.write(f"Function: {function}\n")
        f.write(f"Week: {week}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Samples: {len(y)}\n\n")

        # ---- Model ----
        f.write("MODEL\n")
        f.write("-------------------------------------\n")
        f.write(f"Type: {type(gp).__name__ if gp else 'RF/Hybrid'}\n")
        f.write(f"Kernel: {getattr(gp, 'kernel_', 'N/A')}\n\n")

        # ---- Metrics ----
        f.write("METRICS\n")
        f.write("-------------------------------------\n")
        f.write(f"Best Value: {np.max(y):.6f}\n")
        f.write(f"Mean: {np.mean(y):.6f}\n")
        f.write(f"Std: {np.std(y):.6f}\n\n")

        # ---- Next Point ----
        f.write("NEXT POINT (FOR SIMULATOR)\n")
        f.write("-------------------------------------\n")
        f.write(f"{np.ravel(best_point)}\n")

    # actualizar histórico
    update_global_history(function, week, np.max(y))

    print(f"[REPORT] {path}")

    return path