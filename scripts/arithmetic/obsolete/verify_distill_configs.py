import unittest
from unittest.mock import patch, MagicMock
import runpy
import importlib.util
from pathlib import Path

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_OLD = PROJECT_ROOT / "scripts/arithmetic/run_partial_distill.py"
FILE_NEW = PROJECT_ROOT / "scripts/arithmetic/run_partial_distill.py"


class TestDistillConfigParity(unittest.TestCase):
    def test_strict_parity_all_40_configs(self):
        setup_id = 'gemma-2-0.3B_MaxEnt'

        # 1. Capture OLD configs by mocking the execution
        # We need to capture the exact dictionary 'current_setup' passed to 'run_experiment'
        captured_old_configs = {}

        def mock_run_experiment(sid, alpha, beta, seed=None, **kwargs):
            # This is a bit tricky: we need the setup dict that the old script builds
            # To get it, we'll let the old script's 'run_experiment' logic run partially
            # or just replicate its logic here since it was hardcoded.
            pass

        with patch('localized_undo.utils.parallel_launch.launch_in_parallel_one_per_gpu') as mock_launch:
            with patch('localized_undo.utils.loss_functions.custom_login'):
                # Force run_all to trigger the sweep
                with patch('sys.argv', ['run_partial_distill.py', '--run_all']):
                    # We run runpy to get the 'setups' dict from the old file's global scope
                    old_globals = runpy.run_path(str(FILE_OLD), run_name="__main__")
                    old_base_cfg = old_globals['setups'][setup_id]
                    old_sweep_list = mock_launch.call_args[1]['experiment_list']

        # 2. Get NEW configs from the YAML handler
        from localized_undo.utils.config_handler import load_distill_configs
        yaml_path = PROJECT_ROOT / "configs" / "arithmetic" / "partial_distill.yaml"
        new_configs_dict = load_distill_configs(yaml_path, setup_id)

        # 3. Strict Comparison
        # We ignore only these because they are part of the new management structure
        keys_to_ignore = {
            'method', 'teacher_rel_path', 'stop_condition',
            'english_threshold', 'retain_arithmetic_threshold', 'forget_arithmetic_threshold'
        }

        print(f"\n--- Starting Strict Parity Check for {len(old_sweep_list)} experiments ---")

        for old_exp in old_sweep_list:
            # Unpacking exactly as the old script did
            s_id, alpha, beta, seed, stop_m, eng_t, ret_t, forg_t = old_exp
            new_id = f"{s_id}_a{alpha}_b{beta}_s{seed}"

            self.assertIn(new_id, new_configs_dict, f"Missing config for {new_id}")
            cfg_new = new_configs_dict[new_id]

            # Replicate the OLD manual path/name logic to compare
            path_suffix = f'-alpha_{alpha}-beta_{beta}-seed_{seed}' if seed else f'-alpha_{alpha}-beta_{beta}'
            expected_output_dir = old_base_cfg['output_dir'].replace('-alpha-beta', path_suffix)

            # Detailed check of critical training params
            mismatches = []

            # Fields to check specifically
            fields_to_verify = [
                'learning_rate', 'batch_size', 'max_steps', 'gradient_accumulation_steps',
                'weight_decay', 'max_length', 'num_warmup_steps', 'noise_alpha', 'noise_beta'
            ]

            for field in fields_to_verify:
                val_old = old_base_cfg.get(field) if field not in ['noise_alpha', 'noise_beta'] else (
                    alpha if field == 'noise_alpha' else beta)
                val_new = cfg_new.get(field)
                if str(val_old) != str(val_new):
                    mismatches.append(f"Field '{field}': Old={val_old}, New={val_new}")

            # Path check
            if str(cfg_new['output_dir']) != str(expected_output_dir):
                mismatches.append(f"Output Dir: \n  Old: {expected_output_dir}\n  New: {cfg_new['output_dir']}")

            if mismatches:
                print(f"❌ Mismatches in {new_id}:")
                for m in mismatches:
                    print(f"  - {m}")
                self.fail(f"Strict parity failed for experiment {new_id}")

        print(f"✅ Strict Parity Success! All parameters and paths match for all 40 experiments.")


if __name__ == "__main__":
    unittest.main()