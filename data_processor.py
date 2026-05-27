import numpy as np
import os
import re
from pathlib import Path

class DataCombiner:
    def __init__(self, base_path):
        self.base_path = base_path
        self.initial_data_path = os.path.join(base_path, "initial_data")
        self.outputs_path = os.path.join(base_path, "simulator_data")
        self.combined_data_path = os.path.join(base_path, "combined_data")

    # ---------------------------------------------------------
    # PARSERS
    # ---------------------------------------------------------
    def parse_text_file(self, file_path, data_type="inputs"):
        """Parse text files containing input or output data"""
        try:
            with open(file_path, 'r') as file:
                content = file.read()

            if data_type == "inputs":
                return self._parse_inputs_text(content)
            else:
                return self._parse_outputs_text(content)

        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {str(e)}")
            return None

    # ---------------------------------------------------------
    # ⭐ FINAL INPUT PARSER ⭐
    # ---------------------------------------------------------
    def _parse_inputs_text(self, content):
        """
        Final parser for inputs in the format:
        [array([0.1, 0.2]), array([0.3, 0.4, 0.5]), ...]
        even if everything is in a single line.
        """
        inputs = []

        # Find ALL occurrences of array([...])
        matches = re.findall(r'array\(\s*\[([^\]]+)\]\s*\)', content)

        if not matches:
            print("❌ No valid arrays found in inputs file")
            return []

        for match in matches:
            tokens = re.split(r"[,\s]+", match.strip())
            try:
                nums = [float(t) for t in tokens if t.strip() != ""]
                inputs.append(np.array(nums, dtype=float))
            except:
                continue

        return inputs

    # ---------------------------------------------------------
    # OUTPUT PARSER
    # ---------------------------------------------------------
    def _parse_outputs_text(self, content):
        """Robust parser for outputs in format np.float64(...) or raw numbers"""
        matches = re.findall(r'np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)', content)

        if not matches:
            matches = re.findall(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)

        outputs = [float(m) for m in matches]
        return outputs[:8]

    # ---------------------------------------------------------
    # LOAD INITIAL DATA
    # ---------------------------------------------------------
    def load_initial_data(self):
        initial_inputs = []
        initial_outputs = []

        for i in range(1, 9):
            function_path = os.path.join(self.initial_data_path, f"function_{i}")

            input_file = os.path.join(function_path, "initial_inputs.npy")
            if os.path.exists(input_file):
                initial_inputs.append(np.load(input_file))
                print(f"✅ Loaded: {input_file}")
            else:
                print(f"⚠️ File not found: {input_file}")
                initial_inputs.append(None)

            output_file = os.path.join(function_path, "initial_outputs.npy")
            if os.path.exists(output_file):
                initial_outputs.append(np.load(output_file))
                print(f"✅ Loaded: {output_file}")
            else:
                print(f"⚠️ File not found: {output_file}")
                initial_outputs.append(None)

        return initial_inputs, initial_outputs

    # ---------------------------------------------------------
    # LOAD ONE WEEK
    # ---------------------------------------------------------
    def load_week_data(self, week_num):
        week_path = os.path.join(self.outputs_path, f"Week {week_num}")

        inputs_file = os.path.join(week_path, "inputs.txt")
        week_inputs = self.parse_text_file(inputs_file, "inputs")

        outputs_file = os.path.join(week_path, "outputs.txt")
        week_outputs = self.parse_text_file(outputs_file, "outputs")

        if week_inputs is None or week_outputs is None:
            print(f"❌ Could not load data for Week {week_num}")
            return None, None

        if len(week_inputs) != 8 or len(week_outputs) != 8:
            print(f"⚠️ Week {week_num}: Expected 8 functions, found {len(week_inputs)} inputs and {len(week_outputs)} outputs")

        return week_inputs, week_outputs

    # ---------------------------------------------------------
    # LIST AVAILABLE WEEKS
    # ---------------------------------------------------------
    def get_available_weeks(self):
        if not os.path.exists(self.outputs_path):
            print(f"❌ Outputs folder not found: {self.outputs_path}")
            return []

        weeks = []
        for item in os.listdir(self.outputs_path):
            if item.startswith("Week "):
                try:
                    week_num = int(item.split(" ")[1])
                    weeks.append(week_num)
                except ValueError:
                    continue

        return sorted(weeks)

    # ---------------------------------------------------------
    # LOAD ALL WEEKS UP TO N
    # ---------------------------------------------------------
    def load_all_weeks_up_to(self, target_week):
        all_weeks_data = {'inputs': [], 'outputs': []}

        for week_num in range(1, target_week + 1):
            week_inputs, week_outputs = self.load_week_data(week_num)

            if week_inputs is not None and week_outputs is not None:
                all_weeks_data['inputs'].append((week_num, week_inputs))
                all_weeks_data['outputs'].append((week_num, week_outputs))
                print(f"  📥 Week {week_num} successfully loaded")
            else:
                print(f"  ⚠️ Week {week_num} not available - continuing without it")

        return all_weeks_data

    # ---------------------------------------------------------
    # CUMULATIVE COMBINATION
    # ---------------------------------------------------------
    def combine_data_for_week(self, week_num, initial_inputs, initial_outputs):
        print(f"\n📝 Processing Week {week_num} (CUMULATIVE)...")

        week_output_path = os.path.join(self.combined_data_path, f"week_{week_num}")
        Path(week_output_path).mkdir(parents=True, exist_ok=True)

        all_weeks_data = self.load_all_weeks_up_to(week_num)

        if not all_weeks_data['inputs']:
            print(f"❌ No data found up to Week {week_num}")
            return False

        for i in range(8):
            function_num = i + 1

            combined_inputs_list = []
            combined_outputs_list = []

            # 1. Initial data
            if initial_inputs[i] is not None:
                if initial_inputs[i].ndim == 2:
                    combined_inputs_list.extend(initial_inputs[i])
                else:
                    combined_inputs_list.append(initial_inputs[i])

            if initial_outputs[i] is not None:
                if isinstance(initial_outputs[i], np.ndarray):
                    combined_outputs_list.extend(initial_outputs[i].flatten())
                else:
                    combined_outputs_list.append(float(initial_outputs[i]))

            # 2. Weekly data
            for week_n, week_inputs in all_weeks_data['inputs']:
                combined_inputs_list.append(week_inputs[i])

            for week_n, week_outputs in all_weeks_data['outputs']:
                combined_outputs_list.append(week_outputs[i])

            # 3. Convert to arrays
            final_inputs = np.vstack([inp.reshape(1, -1) if inp.ndim == 1 else inp for inp in combined_inputs_list])
            final_outputs = np.array(combined_outputs_list, dtype=float)

            # 4. FIX: ensure alignment
            if len(final_inputs) != len(final_outputs):
                print(f"⚠️ Mismatch in function {function_num}: inputs={len(final_inputs)}, outputs={len(final_outputs)}")
                min_len = min(len(final_inputs), len(final_outputs))
                final_inputs = final_inputs[:min_len]
                final_outputs = final_outputs[:min_len]
                print(f"   → Automatically trimmed to {min_len}")

            # ---------------------------------------------------------
            # ⭐ DETAILED REPORT PER FUNCTION ⭐
            # ---------------------------------------------------------
            initial_input_count = len(initial_inputs[i]) if initial_inputs[i] is not None else 0
            initial_output_count = len(initial_outputs[i]) if initial_outputs[i] is not None else 0

            new_inputs_added = len(final_inputs) - initial_input_count
            new_outputs_added = len(final_outputs) - initial_output_count

            print(f"\n📘 FUNCTION {function_num} SUMMARY — WEEK {week_num}")
            print(f"   • Initial inputs:      {initial_input_count}")
            print(f"   • Total inputs:        {len(final_inputs)}")
            print(f"   • New inputs added:    {new_inputs_added}")
            print(f"   • Initial outputs:     {initial_output_count}")
            print(f"   • Total outputs:       {len(final_outputs)}")
            print(f"   • New outputs added:   {new_outputs_added}\n")

            # 5. Save
            np.save(os.path.join(week_output_path, f"function_{function_num}_combined_inputs.npy"), final_inputs)
            np.save(os.path.join(week_output_path, f"function_{function_num}_combined_outputs.npy"), final_outputs)

            print(f"  ✅ Function {function_num}: {len(final_inputs)} observations")

        return True

    # ---------------------------------------------------------
    # MAIN ORCHESTRATION
    # ---------------------------------------------------------
    def combine_all_data(self):
        Path(self.combined_data_path).mkdir(parents=True, exist_ok=True)

        print("📂 Loading initial data...")
        initial_inputs, initial_outputs = self.load_initial_data()

        print("\n🔍 Searching for available weeks...")
        available_weeks = self.get_available_weeks()
        print(f"📅 Weeks found: {available_weeks}")

        for week_num in available_weeks:
            self.combine_data_for_week(week_num, initial_inputs, initial_outputs)

        print("\n🎉 Process completed successfully!")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
# PUBLIC API FOR run_week
# ---------------------------------------------------------

# PUBLIC API FOR run_week ✅ NUEVO
# ---------------------------------------------------------
def run_data_pipeline(base_path, week):
    print(f"\n[DATA] Running pipeline for Week {week}")

    combiner = DataCombiner(base_path)

    initial_inputs, initial_outputs = combiner.load_initial_data()

    success = combiner.combine_data_for_week(
        week,
        initial_inputs,
        initial_outputs
    )

    if not success:
        raise ValueError(f"No data available up to Week {week}")

    print(f"[DATA] Week {week} ready ✅")

# ---------------------------------------------------------
# MAIN (solo para testing manual)
# ---------------------------------------------------------
def main():
    base_path = r'C:\Users\JOPOB\JOPOB\15 Imperial_Program\01 Capstone_Project_Imperial_Program\data'

    print("🚀 Cumulative Data Combiner")
    combiner = DataCombiner(base_path)
    combiner.combine_all_data()

if __name__ == "__main__":
    main()
