"""Download models to local HuggingFace cache (no GPU needed).

Models are stored in ~/.cache/huggingface/hub/ by default.
For gated models (e.g. meta-llama/Llama-3.1-8B-Instruct), run `huggingface-cli login` first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download

# Resolve user-facing names to HuggingFace model IDs
_MODEL_IDS = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B-Instruct": "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "zai-org/GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
}

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def main():
    parser = argparse.ArgumentParser(description="Download models to HuggingFace cache")
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help=f"Models to download (default: {DEFAULT_MODELS}). Examples: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3.5-27B, zai-org/GLM-4.7-Flash",
    )
    parser.add_argument("--list", action="store_true", help="List available model IDs and exit")
    parser.add_argument("--all", action="store_true", help="Download all known models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for mid in _MODEL_IDS:
            print(f"  {mid}")
        return

    models = list(_MODEL_IDS) if args.all else args.models
    for model in models:
        model_id = _MODEL_IDS.get(model, model)
        print(f"Downloading {model_id}...")
        try:
            path = snapshot_download(repo_id=model_id)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
