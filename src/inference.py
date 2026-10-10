"""Inference utility for generating responses using the trained LoRA adapter."""

import argparse
import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_pipeline import DEFAULT_SYSTEM_PROMPT, format_chatml
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import get_hardware_info

logger = get_logger("inference")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Generate responses using Fine-Tuned LLM Adapter")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model ID")
    parser.add_argument("--adapter_dir", type=str, default="outputs/final_adapter", help="Directory with trained adapter")
    parser.add_argument("--prompt", type=str, default=None, help="Single instruction prompt to execute")
    parser.add_argument("--context", type=str, default=None, help="Optional context for instruction")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive multi-turn session")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    return parser.parse_args()


def load_model_for_inference(base_model_id: str, adapter_dir: str):
    """Loads base model and attaches the fine-tuned LoRA adapter."""
    hw_info = get_hardware_info()
    device = "cuda" if hw_info["cuda_available"] else "cpu"
    dtype = torch.bfloat16 if (hw_info["cuda_available"] and torch.cuda.is_bf16_supported()) else (
        torch.float16 if hw_info["cuda_available"] else torch.float32
    )

    logger.info(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

    logger.info(f"Loading base model: {base_model_id} on {device} ({dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=False
    )
    if device == "cpu":
        base_model.to("cpu")

    try:
        logger.info(f"Attaching LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        logger.info("LoRA adapter attached successfully.")
    except Exception as e:
        logger.warning(f"Could not load LoRA adapter ({e}). Falling back to base model.")
        model = base_model

    model.eval()
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    context: str = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device: str = "cpu"
) -> str:
    """Formats prompt with ChatML and generates text response."""
    formatted = format_chatml(instruction=prompt, context=context, response="")
    input_text = formatted["prompt"]

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return response


def main() -> None:
    """CLI Entry point for inference."""
    args = parse_args()
    model, tokenizer, device = load_model_for_inference(args.base_model_id, args.adapter_dir)

    if args.interactive:
        print("\n" + "=" * 60)
        print("Interactive Fine-Tuned LLM Chat (type 'quit' or 'exit' to stop)")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nUser > ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break
                if not user_input:
                    continue
                resp = generate_response(
                    model, tokenizer, prompt=user_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device
                )
                print(f"\nAssistant > {resp}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")
    else:
        prompt_text = args.prompt or "Explain how Low-Rank Adaptation (LoRA) reduces trainable parameters in LLMs."
        resp = generate_response(
            model, tokenizer, prompt=prompt_text, context=args.context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device
        )
        print("\n--- Model Response ---")
        print(resp)
        print("----------------------\n")


if __name__ == "__main__":
    main()
