"""End-to-End Supervised Fine-Tuning (SFT) training pipeline with PEFT/LoRA and TRL."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from src.model_utils import apply_lora_adapters, load_model_and_tokenizer
from src.utils.config import load_yaml, merge_configs, validate_training_config
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import get_hardware_info, set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments, allowing hyperparameter overrides."""
    parser = argparse.ArgumentParser(description="QLoRA Supervised Fine-Tuning Pipeline")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to YAML config")
    parser.add_argument("--model_id", type=str, default=None, help="Hugging Face model ID")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (-1 for full epochs)")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha scaling factor")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--final_adapter_dir", type=str, default=None, help="Output directory for final adapter")
    parser.add_argument("--logging_dir", type=str, default=None, help="TensorBoard logging directory")
    parser.add_argument("--report_to", type=str, default=None, help="Tracking platform ('tensorboard' or 'wandb')")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence token length")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads base config and applies CLI argument overrides."""
    base_cfg = load_yaml(args.config)
    overrides: Dict[str, Any] = {}

    if args.model_id:
        overrides.setdefault("model", {})["model_id"] = args.model_id
    if args.lora_r:
        overrides.setdefault("lora", {})["r"] = args.lora_r
    if args.lora_alpha:
        overrides.setdefault("lora", {})["lora_alpha"] = args.lora_alpha

    training_overrides: Dict[str, Any] = {}
    if args.learning_rate is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        training_overrides["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        training_overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.num_train_epochs is not None:
        training_overrides["num_train_epochs"] = args.num_train_epochs
    if args.max_steps is not None:
        training_overrides["max_steps"] = args.max_steps
    if args.output_dir is not None:
        training_overrides["output_dir"] = args.output_dir
    if args.final_adapter_dir is not None:
        training_overrides["final_adapter_dir"] = args.final_adapter_dir
    if args.logging_dir is not None:
        training_overrides["logging_dir"] = args.logging_dir
    if args.report_to is not None:
        training_overrides["report_to"] = args.report_to
    if args.seed is not None:
        training_overrides["seed"] = args.seed

    if training_overrides:
        overrides["training"] = training_overrides

    if args.max_seq_length is not None:
        overrides.setdefault("data", {})["max_seq_length"] = args.max_seq_length

    final_cfg = merge_configs(base_cfg, overrides)
    validate_training_config(final_cfg)
    return final_cfg


def run_training(config: Dict[str, Any]) -> None:
    """Executes the SFTTrainer fine-tuning lifecycle."""
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # 1. Setup seed and inspect hardware
    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    hw_info = get_hardware_info()
    logger.info(f"Hardware Environment: {hw_info}")

    # 2. Verify data splits exist
    train_path = data_cfg["train_path"]
    val_path = data_cfg["validation_path"]
    if not os.path.isfile(train_path) or not os.path.isfile(val_path):
        logger.info(f"Processed splits not found at {train_path}. Triggering data pipeline...")
        from src.data_pipeline import main as run_data_pipeline
        run_data_pipeline()

    logger.info(f"Loading training split from: {train_path}")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    logger.info(f"Loading validation split from: {val_path}")
    val_dataset = load_dataset("json", data_files=val_path, split="train")
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")

    # 3. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_cfg["model_id"],
        quantization_cfg=quant_cfg,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # 4. Inject LoRA Adapters
    is_quantized = quant_cfg.get("load_in_4bit", True) and hw_info["cuda_available"]
    model = apply_lora_adapters(model, lora_cfg, is_quantized=is_quantized)

    # 5. Determine mixed precision based on device support
    use_cuda = hw_info["cuda_available"]
    use_bf16 = train_cfg.get("bf16", True) and use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = train_cfg.get("fp16", False) and use_cuda and not use_bf16

    # 6. Configure TrainingArguments / SFTConfig
    output_dir = train_cfg.get("output_dir", "outputs/checkpoints")
    logging_dir = train_cfg.get("logging_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    report_to = train_cfg.get("report_to", "tensorboard")
    logger.info(f"Configuring MLOps metrics reporting to: '{report_to}' (dir: {logging_dir})")

    training_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "report_to": [report_to] if report_to != "none" else [],
        "learning_rate": float(train_cfg["learning_rate"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 4)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "eval_strategy": train_cfg.get("eval_strategy", "steps"),
        "eval_steps": int(train_cfg.get("eval_steps", 50)),
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "save_steps": int(train_cfg.get("save_steps", 50)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", False)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "seed": seed,
    }

    # Initialize SFTConfig (or TrainingArguments for older trl versions)
    try:
        training_args = SFTConfig(
            max_seq_length=int(data_cfg.get("max_seq_length", 1024)),
            dataset_text_field=data_cfg.get("text_column", "text"),
            **training_kwargs
        )
    except (TypeError, NameError):
        training_args = TrainingArguments(**training_kwargs)

    # 7. Initialize SFTTrainer
    logger.info("Initializing SFTTrainer...")
    sft_init_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "processing_class": tokenizer,
    }
    # Check if SFTConfig handled dataset_text_field or if it needs to be passed directly
    if not isinstance(training_args, SFTConfig):
        sft_init_kwargs["dataset_text_field"] = data_cfg.get("text_column", "text")
        sft_init_kwargs["max_seq_length"] = int(data_cfg.get("max_seq_length", 1024))
        sft_init_kwargs["args"] = training_args
    else:
        sft_init_kwargs["args"] = training_args

    try:
        trainer = SFTTrainer(**sft_init_kwargs)
    except TypeError:
        # Compatibility with older TRL where tokenizer argument is named 'tokenizer'
        if "processing_class" in sft_init_kwargs:
            sft_init_kwargs["tokenizer"] = sft_init_kwargs.pop("processing_class")
        trainer = SFTTrainer(**sft_init_kwargs)

    # 8. Execute Supervised Fine-Tuning
    logger.info("Starting Supervised Fine-Tuning training loop...")
    train_result = trainer.train()
    logger.info(f"Training completed successfully! Global step: {train_result.global_step}, Metrics: {train_result.metrics}")

    # 9. Save Final LoRA Adapter and Tokenizer
    final_adapter_dir = train_cfg.get("final_adapter_dir", "outputs/final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    logger.info(f"Saving final LoRA adapter weights and configuration to: {final_adapter_dir}")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # 10. Rigorous Production Adapter Artifact Validation
    logger.info(f"Validating saved adapter architectural provenance in: {final_adapter_dir}")
    from src.model_utils import validate_saved_adapter
    validation_res = validate_saved_adapter(
        adapter_dir=final_adapter_dir,
        expected_model_id=model_cfg["model_id"],
        lora_cfg=lora_cfg
    )
    logger.info(f"Adapter artifact verification PASSED: {validation_res}")
    logger.info(f"Final model artifacts successfully saved and verified at {final_adapter_dir}!")


def main() -> None:
    """CLI entry point for training."""
    args = parse_args()
    config = build_config_from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()
