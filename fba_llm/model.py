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


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # LLaMA commonly has no pad_token; safe to set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model
