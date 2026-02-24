from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def find_latest_snapshot(cache_root: Path) -> Path:
    snapshots = list(cache_root.glob("models--*/snapshots/*"))
    if not snapshots:
        snapshots = list(cache_root.glob("models--*/*/snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(
            f"Could not find a Hugging Face snapshots folder under {cache_root}"
        )
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def get_input_device(model) -> torch.device:
    """
    Returns a safe device to place input tensors on.
    Works better than model.device when device_map="auto" (sharded models).
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _best_dtype() -> torch.dtype:
    # fp16 on CPU can be problematic; use fp32 if no CUDA
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=_best_dtype(),   # ✅ correct arg name + safer CPU fallback
        low_cpu_mem_usage=True,
    )

    return tokenizer, model