"""Optional utility to permanently merge LoRA adapter weights into base model weights."""

import argparse
import os
import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging_utils import get_logger

logger = get_logger("merge_adapter")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for merging adapters."""
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters into Base LLM")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter_dir", type=str, default="outputs/final_adapter", help="Directory with LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="outputs/merged_model", help="Directory to save merged model")
    parser.add_argument("--torch_dtype", type=str, default="float16", help="Precision for merged weights")
    return parser.parse_args()


def merge_and_save(base_model_id: str, adapter_dir: str, output_dir: str, torch_dtype: str = "float16") -> None:
    """Loads base model in FP16, attaches adapter, merges weights, and exports standalone model.
    
    Args:
        base_model_id: Hugging Face model repository ID.
        adapter_dir: Path to directory containing adapter_config.json and adapter_model.safetensors.
        output_dir: Destination path for merged model.
        torch_dtype: Precision string ('float16' or 'bfloat16').
    """
    logger.info(f"Loading base model '{base_model_id}' in 16-bit precision for merge...")
    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)

    logger.info(f"Loading LoRA adapter from '{adapter_dir}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Executing model.merge_and_unload(). Adding Delta W = (lora_alpha / r) * (B @ A) directly into base weights...")
    merged_model = peft_model.merge_and_unload()

    logger.info(f"Saving merged standalone model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    logger.info("Merge completed successfully! The standalone model is ready for low-latency production deployment.")


def main() -> None:
    """CLI Entry point for merge_adapter."""
    args = parse_args()
    merge_and_save(args.base_model_id, args.adapter_dir, args.output_dir, args.torch_dtype)


if __name__ == "__main__":
    main()
