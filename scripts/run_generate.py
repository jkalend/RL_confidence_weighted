"""Generate synthetic dataset with confidence scores."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DataConfig
from src.data.generate_synthetic import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B-Instruct, Qwen/Qwen3.5-27B, Qwen/Qwen3.5-35B, "
        "meta-llama/Llama-3.1-8B-Instruct, zai-org/GLM-4.7-Flash, openai/gpt-oss-20b. Use Ollama model names (e.g. gpt-oss:20b) with default backend.",
    )
    parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic_target.json")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["unsloth", "ollama", "transformers"],
        default="ollama",
        help="Backend: ollama (default, fast inference), unsloth, or transformers",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output, skip already-processed samples")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        metavar="N",
        help="Ollama context length (num_ctx). Overrides Ollama app setting. E.g. 4096, 8192. Omit to use app default.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config = DataConfig()
    if args.num_ctx is not None:
        config.ollama_num_ctx = args.num_ctx

    generate_synthetic_dataset(
        model_name=args.model,
        config=config,
        output_path=args.output,
        max_samples=args.max_samples,
        backend=args.backend,
        resume=args.resume,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
