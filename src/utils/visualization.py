
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import Config

# ================================================================
# MAIN DISPATCHER
# ================================================================

def generate_plots(function, week, X, y):

    folder = os.path.join(
        Config.PLOTS_BASE,
        function,
        f"week_{week}"
    )
    os.makedirs(folder, exist_ok=True)

    plot_best_curve(function, y, folder)

    dim = X.shape[1]

    if dim == 2:
        plot_2D(function, X, y, folder)

    elif dim <= 6:
        plot_pairplot(function, X, y, folder)

    else:
        plot_projection(function, X, y, folder)


# ================================================================
# INDIVIDUAL PLOTS
# ================================================================
def plot_best_curve(function, y, folder):

    best = np.maximum.accumulate(y)
    plt.figure()
    plt.plot(best)
    plt.title(f"{function} - Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    file = os.path.join(folder, f"{function}_best_curve.png")
    plt.savefig(file)
    plt.close()


def plot_2D(function, X, y, folder):

    plt.figure()
    sc = plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.colorbar(sc)
    plt.title(f"{function} - Observed Points")
    file = os.path.join(folder, f"{function}_2D.png")
    plt.savefig(file)
    plt.close()


def plot_pairplot(function, X, y, folder):

    df = pd.DataFrame(
        X,
        columns=[f"x{i+1}" for i in range(X.shape[1])]
    )
    df["y"] = y

    try:
        df["y_bin"] = pd.qcut(
            df["y"], 4,
            labels=["Q1", "Q2", "Q3", "Q4"]
        )
    except:
        df["y_bin"] = "all"

    sns.pairplot(df, hue="y_bin", corner=True)
    file = os.path.join(folder, f"{function}_pairplot.png")
    plt.savefig(file)
    plt.close()


def plot_projection(function, X, y, folder):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(f"{function} - Projection")
    file = os.path.join(folder, f"{function}_projection.png")
    plt.savefig(file)
    plt.close()
