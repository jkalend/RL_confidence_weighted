"""Run GRPO training with curriculum."""

# Disable torch.compile to avoid Dynamo errors with Qwen3.5 + Unsloth GRPO
# (e.g. apply_rotary_pos_emb shape mismatch, chunked_hidden_states_selective_log_softmax matmul)
import os
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Monkey-patch torch.compile to no-op before any @torch.compile-decorated code loads
import torch  # noqa: E402

def _noop_compile(fn=None, *args, **kwargs):
    if fn is not None:
        return fn
    return lambda f: f

torch.compile = _noop_compile


def _patch_qwen35_rotary_shape_guard() -> None:
    """Guard against Qwen3.5 rotary shape desync under Unsloth GRPO.

    Some Unsloth/TRL forward paths can produce cos/sin sequence length that does not
    match q/k sequence length (including rare zero-length cos/sin), causing:
    `torch.cat([q_embed, q_pass], dim=-1)` to fail in apply_rotary_pos_emb.
    This patch keeps Unsloth enabled and handles the mismatch safely.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_mod
    except Exception:
        return

    original = qwen35_mod.apply_rotary_pos_emb
    if getattr(original, "_rl_confidence_patched", False):
        return

    def _safe_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        try:
            cos_u = cos.unsqueeze(unsqueeze_dim)
            sin_u = sin.unsqueeze(unsqueeze_dim)

            rotary_dim = min(cos_u.shape[-1], q.shape[-1], k.shape[-1])
            if rotary_dim <= 0:
                return q, k

            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

            # Align sequence dimension by trailing window if needed.
            q_len = q_rot.shape[-2]
            cos_len = cos_u.shape[-2]
            if q_len == 0 or cos_len == 0:
                return q, k
            if q_len != cos_len:
                keep = min(q_len, cos_len)
                q_rot = q_rot[..., -keep:, :]
                k_rot = k_rot[..., -keep:, :]
                q_pass = q_pass[..., -keep:, :]
                k_pass = k_pass[..., -keep:, :]
                cos_u = cos_u[..., -keep:, :]
                sin_u = sin_u[..., -keep:, :]

            q_embed = (q_rot * cos_u) + (qwen35_mod.rotate_half(q_rot) * sin_u)
            k_embed = (k_rot * cos_u) + (qwen35_mod.rotate_half(k_rot) * sin_u)

            # If broadcasting produced incompatible leading dims, skip rotary for safety.
            if q_embed.shape[:-1] != q_pass.shape[:-1] or k_embed.shape[:-1] != k_pass.shape[:-1]:
                return q, k

            q_out = torch.cat([q_embed, q_pass], dim=-1)
            k_out = torch.cat([k_embed, k_pass], dim=-1)

            # Re-attach untouched prefix if we had to crop sequence.
            if q_out.shape[-2] != q.shape[-2]:
                prefix = q.shape[-2] - q_out.shape[-2]
                if prefix > 0:
                    q_out = torch.cat([q[..., :prefix, :], q_out], dim=-2)
                    k_out = torch.cat([k[..., :prefix, :], k_out], dim=-2)
            return q_out, k_out
        except Exception:
            return q, k

    _safe_apply_rotary_pos_emb._rl_confidence_patched = True
    qwen35_mod.apply_rotary_pos_emb = _safe_apply_rotary_pos_emb


_patch_qwen35_rotary_shape_guard()

from src.config import Config, DATA_DIR
from src.train import run_grpo_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=DATA_DIR / "synthetic_target.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B",
                        help="Model to use. Default: Qwen/Qwen3.5-4B (5B params). Also: Qwen/Qwen3.5-9B (10B params). "
                             "24 GB VRAM: Qwen3.5-4B, Qwen3.5-9B. Legacy: Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct.")
    parser.add_argument("--epochs", type=int, default=5)
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
