# ================================================================
# RUN WEEK - DATA-DRIVEN BO PIPELINE
# ================================================================
from src.config import Config
from src.optimizer import run_all
from data_processor import run_data_pipeline

# MAIN PIPELINE
def run_week():

    week = Config.CURRENT_WEEK

    print("=" * 60)
    print(f"RUNNING WEEK {week}")
    print("=" * 60)

    # ------------------------------------------------------------
    # STEP 1: DATA
    # ------------------------------------------------------------
    print("\n[STEP 1] Data preparation")

    try:
        run_data_pipeline(Config.BASE_PATH, week)
        print("[OK] Combined data ready")

    except Exception as e:
        print(f"[ERROR] Data step failed: {e}")
        return

    # ------------------------------------------------------------
    # STEP 2: OPTIMIZATION
    # ------------------------------------------------------------
    print("\n[STEP 2] Optimization (data-driven)")

    try:
        # ✅ NO objective functions needed
        run_all()

    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        return

    # ------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------
    print("\n[DONE]")
    print(f"Results stored in: {Config.OUTPUT_BASE}")

    print("\nNext step:")
    print("1. Run simulator with generated inputs")
    print("2. Add results to simulator_data")
    print(f"3. Run Week {week + 1}")


# ================================================================
# ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    run_week()