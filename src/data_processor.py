import numpy as np
import os
import re
from pathlib import Path


# =========================================================
# DATA COMBINER
# =========================================================
class DataCombiner:

    def __init__(self, base_path):

        self.base_path = os.path.abspath(base_path)
        self.initial_data_path = os.path.join(self.base_path, "initial_data")
        self.outputs_path = os.path.join(self.base_path, "simulator_data")
        self.combined_data_path = os.path.join(self.base_path, "combined_data")

        print("\n[DATA PATH CHECK]")
        print(f"Base path       : {self.base_path}")
        print(f"Initial data    : {self.initial_data_path}")
        print(f"Simulator data  : {self.outputs_path}")
        print(f"Combined data   : {self.combined_data_path}")

    # =========================================================
    # PARSERS
    # =========================================================

    def parse_text_file(self, file_path, data_type="inputs"):

        try:
            with open(file_path, 'r') as file:
                content = file.read()

            if data_type == "inputs":
                return self._parse_inputs_text(content)
            else:
                return self._parse_outputs_text(content)

        except FileNotFoundError:
            print(f" File not found: {file_path}")
            return None


    def _parse_inputs_text(self, content):
        inputs = []
        matches = re.findall(r'array\(\s*\[([^\]]+)\]\s*\)', content)

        if not matches:
            print(" No valid arrays found in inputs file")
            return []

        for match in matches:
            tokens = re.split(r"[,\s]+", match.strip())

            try:
                nums = [float(t) for t in tokens if t.strip() != ""]
                inputs.append(np.array(nums, dtype=float))
            except:
                continue

        return inputs


    def _parse_outputs_text(self, content):

        matches = re.findall(
            r'np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)', content
        )

        if not matches:
            matches = re.findall(
                r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content
            )
        return [float(m) for m in matches][:8]

    # =========================================================
    # INITIAL DATA
    # =========================================================

    def load_initial_data(self):

        initial_inputs = []
        initial_outputs = []

        print("\n[LOAD INITIAL DATA]")

        for i in range(1, 9):

            path = os.path.join(self.initial_data_path, f"function_{i}")

            inp_path = os.path.join(path, "initial_inputs.npy")
            out_path = os.path.join(path, "initial_outputs.npy")

            if os.path.exists(inp_path):
                initial_inputs.append(np.load(inp_path))
                print(f" Loaded inputs: {inp_path}")
            else:
                print(f" Missing: {inp_path}")
                initial_inputs.append(None)

            if os.path.exists(out_path):
                initial_outputs.append(np.load(out_path))
                print(f" Loaded outputs: {out_path}")
            else:
                print(f" Missing: {out_path}")
                initial_outputs.append(None)

        return initial_inputs, initial_outputs


    # =========================================================
    # WEEK DATA
    # =========================================================

    def load_week_data(self, week_num):

        week_path = os.path.join(self.outputs_path, f"Week {week_num}")

        inputs_file = os.path.join(week_path, "inputs.txt")
        outputs_file = os.path.join(week_path, "outputs.txt")

        week_inputs = self.parse_text_file(inputs_file, "inputs")
        week_outputs = self.parse_text_file(outputs_file, "outputs")

        if week_inputs is None or week_outputs is None:
            print(f" Could not load data for Week {week_num}")
            return None, None

        return week_inputs, week_outputs

    # =========================================================
    # LOAD ALL WEEKS
    # =========================================================

    def load_all_weeks_up_to(self, target_week):

        all_data = {"inputs": [], "outputs": []}

        for week_num in range(1, target_week + 1):

            week_inputs, week_outputs = self.load_week_data(week_num)

            if week_inputs is not None and week_outputs is not None:
                all_data["inputs"].append((week_num, week_inputs))
                all_data["outputs"].append((week_num, week_outputs))
                print(f" Week {week_num} loaded")
            else:
                print(f" Week {week_num} missing")

        return all_data


    # =========================================================
    # COMBINATION
    # =========================================================

    def combine_data_for_week(self, week_num, initial_inputs, initial_outputs):

        print(f"\n[DATA] Processing Week {week_num} (CUMULATIVE)")

        output_path = os.path.join(self.combined_data_path, f"week_{week_num}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        weeks_data = self.load_all_weeks_up_to(week_num)

        if not weeks_data["inputs"]:
            print(f" No data found up to Week {week_num}")
            return False

        for i in range(8):

            function_name = f"function_{i+1}"

            inputs_list = []
            outputs_list = []

            # ---------------- INITIAL DATA ----------------
            if initial_inputs[i] is not None:
                inputs_list.extend(initial_inputs[i])

            if initial_outputs[i] is not None:
                outputs_list.extend(initial_outputs[i].flatten())

            # ---------------- WEEK DATA ----------------
            for _, data in weeks_data["inputs"]:
                inputs_list.append(data[i])

            for _, data in weeks_data["outputs"]:
                outputs_list.append(data[i])

            # ---------------- BUILD ARRAYS ----------------
            X = np.vstack([
                x.reshape(1, -1) if x.ndim == 1 else x
                for x in inputs_list
            ])

            y = np.array(outputs_list, dtype=float)

            # ---------------- ALIGNMENT FIX ----------------
            if len(X) != len(y):
                n = min(len(X), len(y))
                X, y = X[:n], y[:n]

            # ---------------- SAVE ----------------
            np.save(os.path.join(output_path, f"{function_name}_combined_inputs.npy"), X)
            np.save(os.path.join(output_path, f"{function_name}_combined_outputs.npy"), y)

            print(f"[DATA] {function_name} → samples: {len(y)}")

        return True


def run_data_pipeline(base_path, week):

    print(f"\n[DATA PIPELINE] Week {week}")

    base_path = os.path.abspath(base_path)
    print(f"[PATH] Using base path: {base_path}")

    combiner = DataCombiner(base_path)

    initial_inputs, initial_outputs = combiner.load_initial_data()

    success = combiner.combine_data_for_week(
        week,
        initial_inputs,
        initial_outputs
    )

    if not success:
        raise ValueError(f"No data available up to Week {week}")

    print(f"[DATA] Week {week} ready")



def main():
    from src.config import Config

    print("Cumulative Data Combiner")
    print(f"[PATH] Using data folder: {Config.BASE_PATH}")

    run_data_pipeline(Config.BASE_PATH, Config.CURRENT_WEEK)


if __name__ == "__main__":
    main()