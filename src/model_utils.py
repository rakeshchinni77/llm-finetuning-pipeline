"""Model loading and QLoRA adapter initialization utilities."""

import os
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.utils.logging_utils import get_logger

logger = get_logger("model_utils")

# Allowed parameter size bounds (1B - 7B)
MIN_PARAMS_ALLOWED = 0.8e9
MAX_PARAMS_ALLOWED = 7.5e9


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    compute_dtype: str = "bfloat16"
) -> BitsAndBytesConfig:
    """Constructs Hugging Face BitsAndBytesConfig for 4-bit NormalFloat QLoRA.
    
    Args:
        load_in_4bit: Whether to quantize weights to 4-bit.
        bnb_4bit_quant_type: Quantization datatype ('nf4' or 'fp4').
        bnb_4bit_use_double_quant: Enable nested quantization for memory savings.
        compute_dtype: Computation dtype string ('float16' or 'bfloat16').
        
    Returns:
        Configured BitsAndBytesConfig object.
    """
    dtype = torch.bfloat16 if compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )


def load_model_and_tokenizer(
    model_id: str,
    quantization_cfg: Optional[Dict[str, Any]] = None,
    torch_dtype: str = "bfloat16",
    device_map: Optional[Any] = "auto",
    trust_remote_code: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads pre-trained causal language model (1B-7B) and corresponding tokenizer.
    
    Args:
        model_id: Hugging Face model repository identifier.
        quantization_cfg: Optional dictionary configuring BitsAndBytes 4-bit quantization.
        torch_dtype: Precision string ('float16', 'bfloat16', or 'auto').
        device_map: Accelerate device mapping (e.g. 'auto', 'cpu', or device ID).
        trust_remote_code: Whether to allow remote code execution from HF Hub.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True
    )

    # Configure padding and EOS tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Configured pad_token to match eos_token: '{tokenizer.pad_token}'")
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            logger.info("Added new pad_token: '<|pad|>'")

    # Ensure left padding for causal generation
    tokenizer.padding_side = "left"

    # Setup quantization config if requested and GPU/CUDA is available
    bnb_config = None
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

    if quantization_cfg and quantization_cfg.get("load_in_4bit", True):
        requested_compute = quantization_cfg.get("bnb_4bit_compute_dtype", "float16")
        compute_dtype = "bfloat16" if (requested_compute == "bfloat16" and bf16_supported) else "float16"
        if cuda_available:
            logger.info(f"Configuring 4-bit NormalFloat (NF4) quantization (compute_dtype={compute_dtype})...")
            bnb_config = get_quantization_config(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quantization_cfg.get("bnb_4bit_use_double_quant", True),
                compute_dtype=compute_dtype,
            )
        else:
            logger.warning(
                "CUDA GPU is not available. BitsAndBytes 4-bit quantization requires an NVIDIA GPU. "
                "Falling back to CPU float32/bfloat16 for local development/testing."
            )
            device_map = None

    selected_dtype = torch.bfloat16 if (torch_dtype == "bfloat16" and bf16_supported) else (
        torch.float16 if cuda_available else torch.float32
    )

    logger.info(f"Loading base causal LM '{model_id}' (device_map={device_map}, dtype={selected_dtype})...")
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": selected_dtype,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Align model config precision attributes with selected_dtype to prevent silent BF16 overrides
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.torch_dtype = selected_dtype
        if hasattr(cfg, "get_text_config"):
            txt_cfg = cfg.get_text_config()
            if txt_cfg is not None:
                txt_cfg.torch_dtype = selected_dtype
        hidden_size = getattr(cfg, "hidden_size", None)
        num_layers = getattr(cfg, "num_hidden_layers", None)
        intermediate_size = getattr(cfg, "intermediate_size", None)
        vocab_size = getattr(cfg, "vocab_size", None)
        name_or_path = getattr(cfg, "_name_or_path", model_id)
        logger.info(
            f"Verified Model Architecture: name_or_path='{name_or_path}', "
            f"layers={num_layers}, hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}, vocab_size={vocab_size}, "
            f"config_dtype={cfg.torch_dtype}"
        )

    # Validate parameter count and clarify 4-bit packed representation vs unquantized nominal parameters
    total_elements = sum(p.numel() for p in model.parameters())
    nominal_params = sum(
        p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1)
        for p in model.parameters()
    )
    logger.info(
        f"Model loaded successfully. Base packed elements: {total_elements:,} (~{total_elements / 1e9:.2f}B), "
        f"Nominal unquantized parameters: {nominal_params:,} (~{nominal_params / 1e9:.2f}B)"
    )
    if not (MIN_PARAMS_ALLOWED <= total_elements <= MAX_PARAMS_ALLOWED):
        logger.warning(
            f"Model element count {total_elements / 1e9:.2f}B is outside standard target bounds "
            f"({MIN_PARAMS_ALLOWED/1e9:.1f}B - {MAX_PARAMS_ALLOWED/1e9:.1f}B)."
        )

    return model, tokenizer


def print_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    """Calculates and logs the number of trainable vs. frozen parameters.
    
    Args:
        model: PyTorch model or PeftModel.
        
    Returns:
        Tuple of (trainable_params, all_params, percentage).
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100 * trainable_params / all_param if all_param > 0 else 0.0
    logger.info(
        f"Trainable Parameters: {trainable_params:,} || All Parameters: {all_param:,} || "
        f"Trainable Ratio: {percentage:.4f}%"
    )
    return trainable_params, all_param, percentage


def apply_lora_adapters(
    model: PreTrainedModel,
    lora_cfg: Dict[str, Any],
    is_quantized: bool = True
) -> PeftModel:
    """Prepares quantized model for k-bit training and attaches LoRA adapters.
    
    Args:
        model: Pre-trained base causal language model.
        lora_cfg: Dictionary containing LoRA hyperparameters (r, alpha, target_modules, etc.).
        is_quantized: Whether the model is 4-bit quantized (calls prepare_model_for_kbit_training).
        
    Returns:
        PeftModel with frozen base weights and trainable low-rank adapters.
    """
    logger.info("Initializing PEFT LoRA adapter configuration...")

    if is_quantized and torch.cuda.is_available():
        logger.info("Preparing model for k-bit training (casting norms to float32, enabling gradient checkpointing)...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", True)
        )

    # Explicit LoraConfig specification
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(
        f"LoRA Config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
        f"dropout={peft_config.lora_dropout}, targets={peft_config.target_modules}"
    )

    peft_model = get_peft_model(model, peft_config)

    # Ensure all trainable parameters are strictly in float32 for mixed-precision stability
    # and compatibility with PyTorch GradScaler unscaling on GPUs without native BF16 unscale (Tesla T4).
    trainable_dtypes = set()
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
            trainable_dtypes.add(param.dtype)

    logger.info(f"Verified trainable parameter dtypes: {trainable_dtypes}")
    print_trainable_parameters(peft_model)

    return peft_model


def validate_saved_adapter(
    adapter_dir: str,
    expected_model_id: str,
    lora_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Validates that a saved PEFT adapter directory contains genuine production weights.
    
    Verifies adapter_config.json metadata, safetensors layer coverage, target modules,
    tensor dimensions, and nonzero parameter counts.
    
    Args:
        adapter_dir: Directory containing adapter_config.json and adapter_model.safetensors.
        expected_model_id: The base model ID the adapter was trained for.
        lora_cfg: The configured LoRA hyperparameters dictionary.
        
    Returns:
        Dictionary with validation results summary.
        
    Raises:
        FileNotFoundError: If required adapter files do not exist.
        ValueError: If weights are mock/synthetic, layer coverage is incomplete,
                    or parameter count does not match the production architecture.
    """
    import json
    from pathlib import Path
    from safetensors import safe_open
    from src.utils.audit_utils import get_expected_model_architecture

    adapter_path = Path(adapter_dir)
    safetensors_file = adapter_path / "adapter_model.safetensors"
    config_file = adapter_path / "adapter_config.json"

    if not config_file.is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    if not safetensors_file.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")

    # 1. Validate adapter_config.json
    with open(config_file, "r", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    saved_base = saved_cfg.get("base_model_name_or_path")
    if saved_base != expected_model_id:
        raise ValueError(
            f"Adapter base model mismatch: adapter was configured for '{saved_base}', "
            f"but expected '{expected_model_id}'."
        )

    expected_r = int(lora_cfg.get("r", 16))
    if int(saved_cfg.get("r", -1)) != expected_r:
        raise ValueError(
            f"LoRA rank mismatch: saved adapter r={saved_cfg.get('r')}, expected r={expected_r}."
        )

    expected_targets = set(lora_cfg.get("target_modules", []))
    saved_targets = set(saved_cfg.get("target_modules", []))
    if expected_targets and not expected_targets.issubset(saved_targets):
        missing = expected_targets - saved_targets
        raise ValueError(f"Missing configured target modules in adapter_config.json: {missing}")

    # 2. Validate safetensors tensors
    spec = get_expected_model_architecture(expected_model_id)
    expected_layers_count = spec["num_layers"]
    min_params = spec["min_expected_params"]

    with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"adapter_model.safetensors in {adapter_dir} is empty (0 tensors).")

        total_params = 0
        detected_layers = set()
        detected_modules = set()

        for k in keys:
            tensor = f.get_tensor(k)
            numel = tensor.numel()
            total_params += numel

            # Check that weights are not all zeros or uninitialized NaNs
            if torch.all(tensor == 0):
                raise ValueError(f"Tensor '{k}' contains all zeros, indicating uninitialized adapter weights.")
            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"Tensor '{k}' contains NaN values.")

            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    detected_layers.add(int(parts[i + 1]))
                for tgt in expected_targets:
                    if tgt in p:
                        detected_modules.add(tgt)

    # 3. Check architectural coverage
    if len(detected_layers) < expected_layers_count:
        raise ValueError(
            f"Incomplete layer coverage in adapter: found {len(detected_layers)} layers "
            f"({sorted(list(detected_layers))}), but {expected_model_id} requires {expected_layers_count} layers. "
            f"This indicates a partial or mock model adapter."
        )

    if expected_targets and not expected_targets.issubset(detected_modules):
        missing = expected_targets - detected_modules
        raise ValueError(
            f"Missing target modules in safetensors weights: {missing}. Detected: {detected_modules}"
        )

    if total_params < min_params:
        raise ValueError(
            f"Parameter count {total_params:,} is below minimum threshold of {min_params:,} "
            f"for {expected_model_id} LoRA (r={expected_r}). Possible mock or truncated adapter."
        )

    logger.info(
        f"Adapter validation PASSED: {len(keys)} tensors, {total_params:,} parameters "
        f"across {len(detected_layers)} layers matching {expected_model_id}."
    )
    return {
        "is_valid": True,
        "tensor_count": len(keys),
        "total_parameters": total_params,
        "layers_count": len(detected_layers),
        "target_modules": list(detected_modules),
    }

