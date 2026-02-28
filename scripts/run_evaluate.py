"""Evaluate model on source and target domains."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR
from src.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "final")
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "eval_results.json")
    args = parser.parse_args()

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        max_eval=args.max_eval,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable report (or convert to str)
    out = {k: v for k, v in results.items() if k != "source_report" and k != "target_report"}
    out["source_report"] = str(results.get("source_report", ""))
    out["target_report"] = str(results.get("target_report", ""))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
