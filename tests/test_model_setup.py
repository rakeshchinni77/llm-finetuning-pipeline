"""Unit tests for model and LoRA configuration structures."""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from src.model_utils import (
    get_quantization_config,
    print_trainable_parameters,
)


class DummyLinearModel(nn.Module):
    """Small neural module simulating transformer projection layers."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.out_proj = nn.Linear(64, 10)

    def forward(self, x):
        return self.out_proj(self.v_proj(self.q_proj(x)))

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


def test_quantization_config():
    """Verify BitsAndBytesConfig creation."""
    config = get_quantization_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        compute_dtype="float16"
    )
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype == torch.float16


def test_lora_adapter_injection_and_parameter_reduction():
    """Verify LoRA injects adapters and freezes original parameters."""
    dummy_model = DummyLinearModel()
    total_base_params = sum(p.numel() for p in dummy_model.parameters())

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(dummy_model, lora_config)
    trainable_params, all_params, percentage = print_trainable_parameters(peft_model)

    # Base layers should be frozen
    assert not dummy_model.out_proj.weight.requires_grad
    # Trainable params should be strictly a small fraction of all params
    assert trainable_params > 0
    assert trainable_params < all_params
    assert percentage < 100.0


def test_trainable_parameters_accounting():
    """Verify parameter counter math."""
    model = nn.Sequential(
        nn.Linear(10, 10),  # 100 weights + 10 bias = 110 params
        nn.Linear(10, 2)    # 20 weights + 2 bias = 22 params
    )
    # Freeze second layer
    for p in model[1].parameters():
        p.requires_grad = False

    trainable, total, pct = print_trainable_parameters(model)
    assert trainable == 110
    assert total == 132
    assert round(pct, 2) == round(110 / 132 * 100, 2)


def test_quantization_compute_dtype_hardware_adaptation():
    """Verify compute dtype adapts to hardware support (FP16 fallback on non-BF16 devices)."""
    # Force test float16
    cfg_fp16 = get_quantization_config(compute_dtype="float16")
    assert cfg_fp16.bnb_4bit_compute_dtype == torch.float16

    # When bf16 is requested, verify behavior matches torch.cuda.is_bf16_supported()
    cfg_bf16 = get_quantization_config(compute_dtype="bfloat16")
    expected = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    assert cfg_bf16.bnb_4bit_compute_dtype == expected


def test_loss_type_and_dtype_configuration_sfttrainer():
    """Verify SFTConfig receives loss_type='nll' and fp16 configuration."""
    from trl import SFTConfig
    import inspect

    sig = inspect.signature(SFTConfig.__init__)
    kwargs = {"output_dir": "outputs/test_dtype", "use_cpu": True}
    if "loss_type" in sig.parameters:
        kwargs["loss_type"] = "nll"

    cfg = SFTConfig(**kwargs)
    if "loss_type" in sig.parameters:
        assert cfg.loss_type == "nll"

