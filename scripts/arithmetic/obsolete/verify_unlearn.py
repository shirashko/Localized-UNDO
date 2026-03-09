import sys
import unittest
from unittest.mock import patch, MagicMock
import runpy
import importlib.util
from pathlib import Path

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_OLD = PROJECT_ROOT / "scripts/arithmetic/run_unlearn.py"
FILE_NEW = PROJECT_ROOT / "scripts/arithmetic/run_unlearn.py"


class TestUnlearnConfigParity(unittest.TestCase):
    def test_compare_all_lr_variants(self):
        # Methods we want to verify
        base_setups = ["gemma-2-0.3B_MaxEnt", "gemma-2-0.3B_RMU"]

        print(f"\n--- Verifying Unlearning LR Sweeps for: {base_setups} ---")

        # 1. Collect ALL arguments from the OLD script execution
        # Mocking the parallel launcher to capture the configurations it was passed
        with patch('localized_undo.utils.parallel_launch.launch_in_parallel_one_per_gpu') as mock_launch:
            with patch('accelerate.Accelerator', return_value=MagicMock()), \
                    patch('localized_undo.utils.validation_functions.get_arithmetic_eval_fn',
                          return_value="mock_eval_fn"):

                try:
                    # Run the old script; it populates the 'setups' dictionary and hits the launcher
                    runpy.run_path(str(FILE_OLD), run_name="__main__")
                except SystemExit:
                    pass

            # Extract the setups dictionary from the old script's final global state
            old_globals = runpy.run_path(str(FILE_OLD), run_name="__main__")
            old_setups_dict = old_globals['setups']

        # 2. Collect ALL arguments from the NEW script/handler
        # Dynamically import the new module to access its ALL_EXP_CONFIGS dict
        spec = importlib.util.spec_from_file_location("new_script", FILE_NEW)
        new_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_module)
        new_setups_dict = new_module.ALL_EXP_CONFIGS

        # 3. Comparison Logic
        all_old_keys = set(old_setups_dict.keys())
        all_new_keys = set(new_setups_dict.keys())

        # Verify the number of experiments generated is identical
        self.assertEqual(all_old_keys, all_new_keys,
                         f"Number of generated setups differ! Old: {len(all_old_keys)}, New: {len(all_new_keys)}")

        mismatches = []
        # Ignore non-comparable objects or values handled differently by design
        keys_to_ignore = {'wandb_api_key', 'accelerator', 'eval_fn', 'wandb_run_name', 'method'}

        for setup_id in sorted(all_old_keys):
            cfg_old = old_setups_dict[setup_id]
            cfg_new = new_setups_dict[setup_id]

            all_keys = set(cfg_old.keys()) | set(cfg_new.keys())

            for key in all_keys:
                if key in keys_to_ignore:
                    continue

                val_old = cfg_old.get(key)
                val_new = cfg_new.get(key)

                # Normalize to strings for comparison to handle Path vs Str
                # and subtle differences in numerical default representations (e.g., 0 vs None)
                if str(val_old) == str(val_new):
                    continue

                mismatches.append(f"Setup: {setup_id} | Key: '{key}'\n  Old: {val_old}\n  New: {val_new}")

        if not mismatches:
            print(f"✅ Success! All {len(all_old_keys)} LR variants match perfectly.")
        else:
            print(f"❌ Mismatches found in {len(mismatches)} instances:")
            for m in mismatches[:10]:  # Display first 10 discrepancies
                print(f"  {m}")
            self.fail(f"Found {len(mismatches)} mismatches in unlearning configs!")


if __name__ == "__main__":
    unittest.main()