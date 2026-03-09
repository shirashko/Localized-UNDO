from localized_undo.utils.paths import CONFIG_DIR
from localized_undo.utils.config_handler import load_pretrain_config
from pathlib import Path


def test_pretrain_setup(setup_id):
    print(f"--- Testing Setup: {setup_id} ---")

    yaml_path = CONFIG_DIR / "arithmetic" / "pretrain.yaml"
    try:
        config = load_pretrain_config(yaml_path, setup_id)
        print("✅ Config loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # 2. בדיקת מבנה רשימת הקבצים (התיקון הקריטי)
    train_files = [config['eng_train_file']] + config['secondary_train_files']

    # בדיקה שאין רשימה בתוך רשימה
    is_flat = all(isinstance(item, str) for item in train_files)
    if is_flat:
        print(f"✅ train_files is flat: {train_files}")
    else:
        print(f"❌ ERROR: train_files is nested! Structure: {train_files}")

    missing_files = []
    for f in train_files:
        if not Path(f).exists():
            missing_files.append(f)

    if not missing_files:
        print("✅ All training files exist on disk.")
    else:
        print(f"⚠️ Warning: Missing files: {missing_files}")
        print(f"   (Check if your PROJECT_ROOT is: {Path(__file__).resolve().parent.parent})")

    train_params = {k: v for k, v in config.items() if k not in ['model_id', 'arithmetic_type']}
    print(f"🚀 Params for train(): {list(train_params.keys())}")


if __name__ == "__main__":
    # נסי לבדוק את אחד הסטאפים שהגדרת ב-YAML
    test_pretrain_setup("gemma-2-0.3B_all_arithmetic+eng")