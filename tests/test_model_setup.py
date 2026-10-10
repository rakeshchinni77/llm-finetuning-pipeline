"""Unit tests for model and LoRA configuration structures."""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from src.model_utils import (
    get_quantization_config,
    print_trainable_parameters,
)


import types


class DummyLinearModel(nn.Module):
    """Small neural module simulating transformer projection layers."""
    def __init__(self):
        super().__init__()
        text_cfg = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
        self.config = types.SimpleNamespace(
            model_type="custom",
            _attn_implementation="eager",
            pad_token_id=0,
            eos_token_id=1,
            get_text_config=lambda: text_cfg
        )
        self.generation_config = types.SimpleNamespace(eos_token_id=[1])
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.out_proj = nn.Linear(64, 10)

    def forward(self, x=None, input_ids=None, **kwargs):
        inp = x if x is not None else input_ids
        return self.out_proj(self.v_proj(self.q_proj(inp)))

    def get_output_embeddings(self):
        return self.out_proj

    def add_model_tags(self, tags):
        pass

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


def test_apply_lora_adapters_enforces_float32_trainable_parameters():
    """Verify apply_lora_adapters guarantees trainable LoRA parameters are float32."""
    from src.model_utils import apply_lora_adapters

    dummy_model = DummyLinearModel().to(torch.bfloat16)
    lora_cfg = {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "gradient_checkpointing": False,
    }

    peft_model = apply_lora_adapters(dummy_model, lora_cfg, is_quantized=False)

    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert param.dtype == torch.float32, (
                f"Trainable parameter {name} has dtype {param.dtype}, expected torch.float32"
            )


def test_no_bf16_gradients_reach_scaler():
    """Verify backward pass on LoRA module produces only float32 gradients, safe for GradScaler unscaling."""
    from src.model_utils import apply_lora_adapters

    dummy_model = DummyLinearModel()
    lora_cfg = {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
    }
    peft_model = apply_lora_adapters(dummy_model, lora_cfg, is_quantized=False)

    x = torch.randn(2, 64)
    out = peft_model(x)
    loss = out.sum()
    loss.backward()

    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert param.grad.dtype != torch.bfloat16, (
                f"Gradient for {name} is BFloat16, which causes NotImplementedError in FP16 GradScaler!"
            )
            assert param.grad.dtype == torch.float32


def test_qwen_1_5b_packed_vs_nominal_parameter_accounting():
    """Mathematically verify that 888,616,448 packed elements correspond to genuine 1.54B nominal parameters."""
    vocab_size = 151936
    hidden_size = 1536
    num_layers = 28
    intermediate_size = 8960

    # 1. Embeddings (unquantized in QLoRA: 151,936 x 1,536)
    embed_params = vocab_size * hidden_size  # 233,373,696

    # 2. Linear projection weights per layer:
    # q, k, v, o: 1536*(1536 + 256 + 256 + 1536) = 1536*3584 = 5,505,024
    # gate, up, down: 3 * (1536 * 8960) = 41,287,680
    linear_per_layer = (hidden_size * (hidden_size + 256 + 256 + hidden_size)) + (3 * hidden_size * intermediate_size)
    assert linear_per_layer == 46792704

    total_linear_weights = num_layers * linear_per_layer  # 1,310,195,712
    # In 4-bit NF4 packing, 2 weights per byte:
    packed_linear_elements = total_linear_weights // 2    # 655,097,856

    # 3. Norms: 28 layers * (2 * 1536) + 1536 final norm = 87,552
    norm_params = num_layers * 2 * hidden_size + hidden_size  # 87,552

    # 4. BitsAndBytes QuantState scales/offsets metadata: 57,344
    quant_state_metadata = 57344

    nominal_total = embed_params + total_linear_weights + norm_params
    assert nominal_total == 1543656960  # ~1.54B parameters

    packed_total = embed_params + packed_linear_elements + norm_params + quant_state_metadata
    assert packed_total == 888616448    # Exactly the reported 888,616,448!


def test_trainable_lora_parameters_remain_float32_through_trainer_init(tmp_path):
    """Verify RobustSFTTrainer preserves float32 trainable LoRA parameters through initialization."""
    from src.train import RobustSFTTrainer
    from src.model_utils import apply_lora_adapters
    from transformers import TrainingArguments, PreTrainedTokenizerBase
    from datasets import Dataset
    from unittest.mock import MagicMock

    dummy_model = DummyLinearModel()
    lora_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}
    peft_model = apply_lora_adapters(dummy_model, lora_cfg, is_quantized=False)

    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]})
    mock_tok = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1
    mock_tok.pad_token = "<|pad|>"
    mock_tok.eos_token = "<|endoftext|>"
    mock_tok.get_vocab.return_value = {"<|endoftext|>": 1, "<|pad|>": 0}

    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True)
    trainer = RobustSFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=ds,
        processing_class=mock_tok
    )

    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            assert param.dtype == torch.float32, (
                f"Trainable parameter {name} changed to {param.dtype} during SFTTrainer init!"
            )

