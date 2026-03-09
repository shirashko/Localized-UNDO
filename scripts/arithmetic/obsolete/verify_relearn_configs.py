import unittest
from unittest.mock import patch, MagicMock
import runpy
from pathlib import Path

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_OLD = PROJECT_ROOT / "scripts/arithmetic/run_relearn.py"


class TestRelearnConfigParity(unittest.TestCase):
    def test_strict_relearn_parity(self):
        target_setup = "gemma-2-0.3B_all_data"
        print(f"\n--- Starting Strict Relearning Parity Check (Modern Structure) ---")

        # 1. Load the OLD script's globals
        with patch('localized_undo.utils.parallel_launch.launch_in_parallel_one_per_gpu'):
            with patch('localized_undo.utils.loss_functions.custom_login'):
                old_globals = runpy.run_path(str(FILE_OLD), run_name="__main__")

        old_setups_dict = old_globals['setups']
        old_models_list = old_globals['MODELS_TO_RUN']
        old_lrs_list = old_globals['LRS_TO_RUN']
        old_setups_to_run = old_globals['SETUPS_TO_RUN']

        old_experiment_params = [
            (setup_id, lr, model)
            for setup_id in old_setups_to_run
            for lr in old_lrs_list
            for model in old_models_list
        ]

        # 2. Get NEW configs from the YAML handler
        from localized_undo.utils.config_handler import load_relearn_configs
        from localized_undo.utils.paths import CONFIG_DIR

        yaml_path = CONFIG_DIR / "arithmetic" / "relearn.yaml"
        new_configs_dict = load_relearn_configs(yaml_path, [target_setup], old_models_list)

        # 3. Comparison Logic
        print(f"Total combinations to verify for {target_setup}: {len(old_lrs_list) * len(old_models_list)}")

        mismatches = []

        for setup_id, lr, model_path in old_experiment_params:
            if setup_id != target_setup:
                continue

            safe_model_name = model_path.replace('/', '_')
            unique_id = f"{setup_id}_{safe_model_name}_lr{float(lr)}"

            self.assertIn(unique_id, new_configs_dict, f"Missing config for {unique_id}")
            cfg_new = new_configs_dict[unique_id]

            # --- Verification of Key Logic (Ignoring purely cosmetic string diffs) ---

            # A. Check Learning Rate (Numerical parity)
            if float(cfg_new['learning_rate']) != float(lr):
                mismatches.append(f"[{unique_id}] LR Mismatch: Old={lr}, New={cfg_new['learning_rate']}")

            # B. Check Model Source Path (Logic parity)
            expected_model_name = old_setups_dict[setup_id]['model_name'].replace('model_path', model_path)
            if "distilled" not in expected_model_name and "pretrained" not in expected_model_name:
                expected_model_name = expected_model_name.replace('final_student', "final_model")

            if str(cfg_new['model_name']) != str(expected_model_name):
                mismatches.append(f"[{unique_id}] Source Model Path Mismatch")

            # C. Verify the Modern Output Directory Structure
            new_path = str(cfg_new['output_dir'])
            lr_scientific = f"{float(lr):.1e}"

            # Check for middle folder (setup_id)
            if setup_id not in new_path:
                mismatches.append(f"[{unique_id}] Modern path missing setup_id folder: {setup_id}")

            # Check for scientific notation in label
            if lr_scientific not in new_path:
                mismatches.append(f"[{unique_id}] Modern path missing scientific LR: {lr_scientific}")

        if not mismatches:
            print(f"✅ Success! Modern configuration logic matches research intent.")
        else:
            print(f"❌ Found {len(mismatches)} logic mismatches:")
            for m in mismatches[:3]:
                print(f"  {m}")
            self.fail(f"Strict parity failed for {len(mismatches)} relearning configs.")


if __name__ == "__main__":
    unittest.main()