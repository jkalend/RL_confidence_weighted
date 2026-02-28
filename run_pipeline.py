"""Full pipeline: generate -> train -> evaluate."""

import argparse
import json
from pathlib import Path

from src.config import Config, DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR
from src.data.generate_synthetic import generate_synthetic_dataset
from src.train import run_grpo_training, load_synthetic_for_training
from src.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true", help="Use existing synthetic data")
    parser.add_argument("--skip-train", action="store_true", help="Use existing checkpoint")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", type=str, choices=["unsloth", "ollama", "transformers"], default="unsloth")
    args = parser.parse_args()

    # 1. Generate synthetic data
    synthetic_path = DATA_DIR / "synthetic_target.json"
    if not args.skip_generate:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_synthetic_dataset(
            model_name=args.model,
            output_path=synthetic_path,
            max_samples=args.max_samples,
            backend=args.backend,
        )
    elif not synthetic_path.exists():
        raise FileNotFoundError(f"Run without --skip-generate first or provide {synthetic_path}")

    # 2. Train
    if not args.skip_train:
        config = Config()
        config.model.model_name = args.model
        config.grpo.num_train_epochs = args.epochs
        run_grpo_training(
            config=config,
            synthetic_path=synthetic_path,
            use_unsloth=(args.backend == "unsloth"),
        )
    else:
        # Check if checkpoint exists before skipping to evaluation
        checkpoint_path = CHECKPOINT_DIR / "final"
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("Please run training first (remove --skip-train) or provide a checkpoint.")
            exit(1)

    # 3. Evaluate
    results = run_evaluation(
        checkpoint_path=CHECKPOINT_DIR / "final",
        max_eval=min(100, args.max_samples),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "pipeline_results.json"
    out = {k: v for k, v in results.items() if isinstance(v, (int, float, str, bool, type(None)))}
    out["source_report"] = str(results.get("source_report", ""))
    out["target_report"] = str(results.get("target_report", ""))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
