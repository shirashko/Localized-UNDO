import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CACHE_DIR = PROJECT_ROOT / ".cache"
DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models" / "non-wmdp"
LOCALIZATION_MASKS_DIR = PROJECT_ROOT / "localization_masks"
WMDP_MODEL_DIR = PROJECT_ROOT / "models" / "wmdp"
CONFIG_DIR = PROJECT_ROOT / "configs"
TOKENS_DIR = PROJECT_ROOT / "tokens"

WANDB_API_KEY_PATH = TOKENS_DIR / "wandb_token.txt"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(CACHE_DIR)