"""Run GRPO training with curriculum."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, DATA_DIR
from src.train import run_grpo_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=DATA_DIR / "synthetic_target.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use (e.g., Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B-Instruct, meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--no-unsloth", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.model.model_name = args.model
    config.grpo.num_train_epochs = args.epochs

    run_grpo_training(
        config=config,
        synthetic_path=args.synthetic,
        use_unsloth=not args.no_unsloth,
    )


if __name__ == "__main__":
    main()
