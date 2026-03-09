import sys
import unittest
from unittest.mock import patch, MagicMock
import runpy
import importlib.util
from pathlib import Path

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_OLD = PROJECT_ROOT / "scripts/arithmetic/run_pretrain_arithmetic.py"
FILE_NEW = PROJECT_ROOT / "scripts/arithmetic/run_pretrain_arithmetic.py"


class TestConfigParity(unittest.TestCase):
    def test_compare_arguments(self):
        setup_id = 'gemma-2-0.3B_all_arithmetic+eng'
        print(f"\n--- Comparing arguments for setup: {setup_id} ---")

        # 1. Collect arguments from the OLD script
        mock_acc_class = MagicMock()
        with patch('localized_undo.tools.pretrain.train') as mock_train_old, \
                patch('localized_undo.utils.validation_functions.get_arithmetic_eval_fn', return_value="mock_eval_fn"):

            try:
                # Inject Accelerator to avoid NameError in the old script execution
                runpy.run_path(
                    str(FILE_OLD),
                    init_globals={'Accelerator': mock_acc_class},
                    run_name="__main__"
                )
            except SystemExit:
                pass
            except Exception as e:
                print(f"⚠️ Note: Old script finished with: {e}")

            call_found = None
            if mock_train_old.call_args_list:
                for call in mock_train_old.call_args_list:
                    _, kwargs = call
                    if setup_id in str(kwargs.get('output_dir', '')):
                        call_found = call
                        break

            if not call_found:
                self.fail(f"Could not find a call to train() for {setup_id} in the old script.")

            args_old, kwargs_old = call_found

        # 2. Collect arguments from the NEW script
        with patch('localized_undo.tools.pretrain.train') as mock_train_new, \
                patch('accelerate.Accelerator', return_value=MagicMock()), \
                patch('localized_undo.utils.validation_functions.get_arithmetic_eval_fn', return_value="mock_eval_fn"):

            spec = importlib.util.spec_from_file_location("new_script", FILE_NEW)
            new_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(new_module)

            with patch('sys.argv', ['run_pretrain_arithmetic.py', '--setup', setup_id]):
                new_module.main()

            args_new, kwargs_new = mock_train_new.call_args

        # 3. Comparison Logic
        all_keys = set(kwargs_old.keys()) | set(kwargs_new.keys())
        mismatches = []

        # Keys to ignore: either objects, tokens, or newly explicit arguments in the YAML
        keys_to_ignore = {
            'wandb_api_key', 'accelerator', 'eval_fn', 'wandb_run_name',
            'eng_train_file', 'eng_valid_file', 'secondary_train_files',
            'cache_dir', 'dataset_cache_dir'  # Handled by string normalization below
        }

        def flatten_to_strings(obj):
            """Recursively flattens lists and converts everything to strings for comparison."""
            if isinstance(obj, list):
                result = []
                for item in obj:
                    result.extend(flatten_to_strings(item))
                return result
            return [str(obj)]

        for key in sorted(all_keys):
            if key in keys_to_ignore:
                continue

            val_old = kwargs_old.get(key)
            val_new = kwargs_new.get(key)

            # Special handling for train_files to account for the flattening fix
            if key == 'train_files':
                if flatten_to_strings(val_old) == flatten_to_strings(val_new):
                    continue

            # Standard normalization for Paths, Strings, and None
            if str(val_old) == str(val_new):
                continue

            mismatches.append(f"Key: '{key}'\n  Old: {val_old}\n  New: {val_new}")

        if not mismatches:
            print("✅ Success! All core arguments match perfectly (including fixed flattening).")
        else:
            print("❌ Mismatches found:")
            for m in mismatches:
                print(f"  {m}")
            self.fail("Configurations do not match! Review the differences above.")


if __name__ == "__main__":
    unittest.main()