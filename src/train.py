"""End-to-End Supervised Fine-Tuning (SFT) training pipeline with PEFT/LoRA and TRL."""

import argparse
import inspect
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
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable FP16")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable BF16")
    parser.add_argument("--loss_type", type=str, default=None, help="Loss computation type ('nll' or 'chunked_nll')")
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
    if args.fp16 is not None:
        training_overrides["fp16"] = args.fp16
    if args.bf16 is not None:
        training_overrides["bf16"] = args.bf16
    if args.loss_type is not None:
        training_overrides["loss_type"] = args.loss_type

    if training_overrides:
        overrides["training"] = training_overrides

    if args.max_seq_length is not None:
        overrides.setdefault("data", {})["max_seq_length"] = args.max_seq_length

    final_cfg = merge_configs(base_cfg, overrides)
    validate_training_config(final_cfg)
    return final_cfg


class RobustSFTTrainer(SFTTrainer):
    """SFTTrainer with hardware-adaptive fallback for devices without chunked BF16 GEMM support."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        except (RuntimeError, AttributeError) as exc:
            err_msg = str(exc)
            if any(k in err_msg for k in ["BFloat16", "apply", "compute capability", "_ChunkedLogProbFunction"]):
                labels = inputs.get("labels")
                logits_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "fused_lm_head")}
                outputs = model(**logits_inputs)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return (loss, outputs) if return_outputs else loss
            raise exc

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Performs a training step with diagnostic logging after backward and before gradient unscaling."""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if not getattr(self, "_diagnostic_logged", False):
            self._diagnostic_logged = True
            param_dtypes = {}
            grad_dtypes = {}
            has_bf16_grad = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    p_dt = str(param.dtype)
                    param_dtypes[p_dt] = param_dtypes.get(p_dt, 0) + 1
                    if param.grad is not None:
                        g_dt = str(param.grad.dtype)
                        grad_dtypes[g_dt] = grad_dtypes.get(g_dt, 0) + 1
                        if param.grad.dtype == torch.bfloat16:
                            has_bf16_grad = True

            logger.info(f"[DIAGNOSTIC] Trainable parameter dtypes: {param_dtypes}")
            logger.info(f"[DIAGNOSTIC] Trainable gradient dtypes: {grad_dtypes}")
            logger.info(f"[DIAGNOSTIC] Any BFloat16 trainable gradients detected: {has_bf16_grad}")
            if has_bf16_grad:
                logger.error(
                    "[DIAGNOSTIC ERROR] BFloat16 gradient detected on trainable parameter! "
                    "This causes NotImplementedError in _amp_foreach_non_finite_check_and_unscale_cuda."
                )

        return loss


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

    # Guarantee SFTTrainer prompt-completion schema compatibility
    if "completion" not in train_dataset.column_names:
        logger.info("Injecting missing 'completion' field into train split from ground-truth 'response'...")
        train_dataset = train_dataset.map(lambda ex: {"completion": f"{ex['response'].strip()}<|im_end|>"})
    if "completion" not in val_dataset.column_names:
        logger.info("Injecting missing 'completion' field into validation split from ground-truth 'response'...")
        val_dataset = val_dataset.map(lambda ex: {"completion": f"{ex['response'].strip()}<|im_end|>"})

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
    bf16_supported = use_cuda and torch.cuda.is_bf16_supported()

    configured_bf16 = bool(train_cfg.get("bf16", False))
    configured_fp16 = bool(train_cfg.get("fp16", True))

    if use_cuda:
        if bf16_supported and configured_bf16:
            use_bf16 = True
            use_fp16 = False
            logger.info("Hardware supports BF16 (compute capability >= 8.0). Enabled bf16=True.")
        else:
            use_bf16 = False
            use_fp16 = True
            if configured_bf16 and not bf16_supported:
                logger.warning(
                    f"BF16 requested but GPU '{hw_info.get('device_name')}' does not support BF16 GEMM "
                    f"(compute capability < 8.0). Automatically falling back to FP16 mixed precision."
                )
            else:
                logger.info("Using FP16 mixed precision for CUDA device.")
    else:
        use_bf16 = False
        use_fp16 = False
        logger.info("CUDA not available. CPU training will use float32.")

    loss_type = train_cfg.get("loss_type", "nll")
    logger.info(f"Configuring SFT loss function type: '{loss_type}'")

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
        "loss_type": loss_type,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "seed": seed,
    }

    # Dynamically filter training kwargs based on supported parameters in the installed TRL/Transformers
    sig = inspect.signature(SFTConfig.__init__)
    supported_params = set(sig.parameters.keys())

    clean_kwargs = {k: v for k, v in training_kwargs.items() if k in supported_params and k != "logging_dir"}
    if not use_cuda and "use_cpu" in supported_params:
        clean_kwargs["use_cpu"] = True

    # Handle warmup_steps if warmup_ratio is unsupported
    if "warmup_steps" in supported_params and "warmup_ratio" not in supported_params:
        max_steps = int(training_kwargs.get("max_steps", 60))
        warmup_ratio = float(training_kwargs.get("warmup_ratio", 0.03))
        clean_kwargs["warmup_steps"] = max(1, int(max_steps * warmup_ratio)) if max_steps > 0 else 5

    max_len = int(data_cfg.get("max_seq_length", 1024))
    dataset_text_field = data_cfg.get("text_column", "text")

    if "max_length" in supported_params:
        clean_kwargs["max_length"] = max_len
    elif "max_seq_length" in supported_params:
        clean_kwargs["max_seq_length"] = max_len

    if "dataset_text_field" in supported_params:
        clean_kwargs["dataset_text_field"] = dataset_text_field

    try:
        training_args = SFTConfig(**clean_kwargs)
    except Exception as e:
        logger.warning(f"SFTConfig instantiation fell back due to: {e}. Trying TrainingArguments fallback...")
        ta_sig = inspect.signature(TrainingArguments.__init__)
        ta_params = set(ta_sig.parameters.keys())
        ta_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params and k != "logging_dir"}
        if "warmup_steps" in ta_params and "warmup_ratio" not in ta_params:
            ta_kwargs["warmup_steps"] = max(1, int(int(training_kwargs.get("max_steps", 60)) * float(training_kwargs.get("warmup_ratio", 0.03))))
        if not use_cuda and "use_cpu" in ta_params:
            ta_kwargs["use_cpu"] = True
        training_args = TrainingArguments(**ta_kwargs)

    try:
        training_args.logging_dir = logging_dir
    except Exception:
        pass

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
        trainer = RobustSFTTrainer(**sft_init_kwargs)
    except TypeError:
        # Compatibility with older TRL where tokenizer argument is named 'tokenizer'
        if "processing_class" in sft_init_kwargs:
            sft_init_kwargs["tokenizer"] = sft_init_kwargs.pop("processing_class")
        trainer = RobustSFTTrainer(**sft_init_kwargs)

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
