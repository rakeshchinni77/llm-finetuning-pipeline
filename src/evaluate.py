"""Comparative evaluation script: Base Model vs. Fine-Tuned PEFT/LoRA Model on held-out test set."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.utils.config import load_yaml, merge_configs, validate_evaluation_config
from src.utils.logging_utils import get_logger
from src.utils.metrics import compute_perplexity, compute_rouge_scores
from src.utils.reproducibility import get_hardware_info, set_seed

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned LLM")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml", help="Path to evaluation config")
    parser.add_argument("--base_model_id", type=str, default=None, help="Base model identifier")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to held-out test.jsonl")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max tokens to generate per response")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    return parser.parse_args()


def load_test_dataset(test_path: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Loads held-out test dataset from JSONL.
    
    Args:
        test_path: Path to test.jsonl.
        max_samples: Optional limit on evaluation count.
        
    Returns:
        List of dictionaries with 'instruction', 'context', 'response', 'prompt'.
    """
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    records: List[Dict[str, str]] = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples and max_samples > 0:
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} test examples from {test_path}")
    return records


def generate_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    gen_config: Dict[str, Any],
    device: str
) -> str:
    """Generates autoregressive response for a single prompt using greedy or sampling decoding."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 256),
            do_sample=gen_config.get("do_sample", False),
            temperature=gen_config.get("temperature", 0.1) if gen_config.get("do_sample", False) else 1.0,
            top_p=gen_config.get("top_p", 0.9) if gen_config.get("do_sample", False) else 1.0,
            repetition_penalty=gen_config.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the input prompt tokens to extract only the generated response
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def run_evaluation(config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs dual evaluation comparing Base Model vs. Fine-Tuned Model."""
    model_cfg = config["model"]
    data_cfg = config["data"]
    gen_cfg = config["generation"]
    metrics_cfg = config["metrics"]
    out_cfg = config["output"]

    base_model_id = model_cfg["base_model_id"]
    adapter_dir = model_cfg["adapter_dir"]
    test_path = data_cfg["test_data_path"]
    max_samples = data_cfg.get("max_eval_samples", 50)

    # Load test set
    test_data = load_test_dataset(test_path, max_samples=max_samples)
    prompts = [item["prompt"] for item in test_data]
    references = [item["response"] for item in test_data]

    # Hardware detection
    hw_info = get_hardware_info()
    device = "cuda" if hw_info["cuda_available"] else "cpu"
    dtype = torch.bfloat16 if (hw_info["cuda_available"] and torch.cuda.is_bf16_supported()) else (
        torch.float16 if hw_info["cuda_available"] else torch.float32
    )

    logger.info(f"Evaluation Runtime Device: {device} (dtype: {dtype})")

    # Load Tokenizer
    logger.info(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    tokenizer.padding_side = "left"

    # Step 1: Base Model Generation
    logger.info(f"--- [1/2] Evaluating Base Model: {base_model_id} ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=False
    )
    if device == "cpu":
        base_model.to("cpu")
    base_model.eval()

    base_predictions: List[str] = []
    for idx, prompt in enumerate(prompts):
        logger.info(f"Base model generating sample {idx + 1}/{len(prompts)}...")
        pred = generate_prediction(base_model, tokenizer, prompt, gen_cfg, device)
        base_predictions.append(pred)

    base_rouge = compute_rouge_scores(base_predictions, references) if metrics_cfg.get("compute_rouge", True) else {}
    base_ppl = compute_perplexity(base_model, tokenizer, references, device=device) if metrics_cfg.get("compute_perplexity", True) else None
    logger.info(f"Base Model Scores: ROUGE={base_rouge}, Perplexity={base_ppl}")

    # Step 2: Fine-Tuned Model Generation (Base + LoRA Adapter)
    logger.info(f"--- [2/2] Evaluating Fine-Tuned PEFT Model (Adapter: {adapter_dir}) ---")
    adapter_exists = os.path.isdir(adapter_dir) and (
        os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
    )

    tuned_predictions: List[str] = []
    if adapter_exists:
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        peft_model.eval()
        for idx, prompt in enumerate(prompts):
            logger.info(f"Fine-tuned model generating sample {idx + 1}/{len(prompts)}...")
            pred = generate_prediction(peft_model, tokenizer, prompt, gen_cfg, device)
            tuned_predictions.append(pred)

        tuned_rouge = compute_rouge_scores(tuned_predictions, references) if metrics_cfg.get("compute_rouge", True) else {}
        tuned_ppl = compute_perplexity(peft_model, tokenizer, references, device=device) if metrics_cfg.get("compute_perplexity", True) else None
        logger.info(f"Fine-Tuned Model Scores: ROUGE={tuned_rouge}, Perplexity={tuned_ppl}")
    else:
        logger.warning(
            f"Adapter directory '{adapter_dir}' not found or unpopulated! "
            f"Please run training first (Phase 05) to generate trained adapter weights."
        )
        tuned_predictions = ["<adapter_not_trained>" for _ in base_predictions]
        tuned_rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        tuned_ppl = float("nan")

    # Consolidate results
    comparison_summary = {
        "evaluation_dataset_size": len(test_data),
        "base_model": {
            "model_id": base_model_id,
            "rouge1": base_rouge.get("rouge1", 0.0),
            "rouge2": base_rouge.get("rouge2", 0.0),
            "rougeL": base_rouge.get("rougeL", 0.0),
            "perplexity": base_ppl,
        },
        "fine_tuned_model": {
            "model_id": base_model_id,
            "adapter_dir": adapter_dir,
            "rouge1": tuned_rouge.get("rouge1", 0.0),
            "rouge2": tuned_rouge.get("rouge2", 0.0),
            "rougeL": tuned_rouge.get("rougeL", 0.0),
            "perplexity": tuned_ppl,
        },
        "improvements": {
            "delta_rouge1": round(tuned_rouge.get("rouge1", 0.0) - base_rouge.get("rouge1", 0.0), 4),
            "delta_rouge2": round(tuned_rouge.get("rouge2", 0.0) - base_rouge.get("rouge2", 0.0), 4),
            "delta_rougeL": round(tuned_rouge.get("rougeL", 0.0) - base_rouge.get("rougeL", 0.0), 4),
            "ppl_reduction": round(base_ppl - tuned_ppl, 4) if (base_ppl and tuned_ppl and not pd.isna(tuned_ppl)) else None,
        }
    }

    # Build qualitative comparison dataframe
    df_rows = []
    for i in range(len(test_data)):
        df_rows.append({
            "sample_id": i + 1,
            "instruction": test_data[i]["instruction"],
            "context": test_data[i].get("context", ""),
            "ground_truth": references[i],
            "base_model_response": base_predictions[i],
            "fine_tuned_response": tuned_predictions[i],
        })
    df_results = pd.DataFrame(df_rows)

    # Save outputs
    predictions_dir = Path(out_cfg.get("predictions_dir", "outputs/predictions"))
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = out_cfg.get("results_json", "outputs/predictions/evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info(f"Saved quantitative metrics to {metrics_json_path}")

    csv_path = out_cfg.get("comparison_csv", "outputs/predictions/comparison_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved detailed side-by-side predictions to {csv_path}")

    return comparison_summary, df_results


def main() -> None:
    """CLI Entry point for evaluate."""
    args = parse_args()
    base_cfg = load_yaml(args.config)

    overrides: Dict[str, Any] = {}
    if args.base_model_id:
        overrides.setdefault("model", {})["base_model_id"] = args.base_model_id
    if args.adapter_dir:
        overrides.setdefault("model", {})["adapter_dir"] = args.adapter_dir
    if args.test_data_path:
        overrides.setdefault("data", {})["test_data_path"] = args.test_data_path
    if args.max_eval_samples:
        overrides.setdefault("data", {})["max_eval_samples"] = args.max_eval_samples
    if args.max_new_tokens:
        overrides.setdefault("generation", {})["max_new_tokens"] = args.max_new_tokens

    final_cfg = merge_configs(base_cfg, overrides)
    validate_evaluation_config(final_cfg)
    set_seed(args.seed)

    summary, _ = run_evaluation(final_cfg)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS COMPARISON SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
