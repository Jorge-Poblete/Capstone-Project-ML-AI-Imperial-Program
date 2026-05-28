# ================================================================
# RUN WEEK - DATA-DRIVEN BO PIPELINE
# ================================================================

from src.config import Config
from src.optimizer import run_all
from src.data_processor import run_data_pipeline


# ================================================================
# MAIN PIPELINE
# ================================================================

def run_week():

    week = Config.CURRENT_WEEK

    print("\n" + "=" * 60)
    print(f"[RUN WEEK] Week {week}")
    print("=" * 60)

    print("\n[PATH CHECK]")
    print(f"Base path   : {Config.BASE_PATH}")
    print(f"Output path : {Config.OUTPUT_BASE}")

    # ------------------------------------------------------------
    # STEP 1: DATA PREPARATION
    # ------------------------------------------------------------
    print("\n[STEP 1] DATA PIPELINE")

    try:
        run_data_pipeline(Config.BASE_PATH, week)
        print("[SUCCESS] Data pipeline completed")

    except Exception as e:
        print(f"[ERROR] Data pipeline failed:\n{e}")
        print("[STOP] Cannot continue without data")
        return

    # ------------------------------------------------------------
    # STEP 2: OPTIMIZATION
    # ------------------------------------------------------------
    print("\n[STEP 2] OPTIMIZATION ENGINE")

    try:
        run_all()
        print("[SUCCESS] Optimization completed")

    except Exception as e:
        print(f"[ERROR] Optimization failed:\n{e}")
        return

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[DONE] WEEK COMPLETED")
    print("=" * 60)

    print(f"\nOutputs stored at:\n{Config.OUTPUT_BASE}")

    print("\nNEXT STEPS:")
    print("  1. Review reports in /outputs/reports/")
    print(f"  2. Run simulator using suggested next points (from reports)")
    print(f"  3. Save results in: data/simulator_data/Week {week + 1}")
    print(f"  4. Update Config.CURRENT_WEEK = {week + 1}")
    print("  5. Run this script again\n")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    print(" Starting Bayesian Optimization Pipeline...")

    run_week()
