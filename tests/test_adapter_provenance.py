"""Regression tests for adapter provenance, architectural verification, and test artifact isolation."""

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

from src.model_utils import validate_saved_adapter


def test_validator_rejects_mock_or_truncated_adapter(tmp_path):
    """Verify that validate_saved_adapter rejects a tiny/mock 2-tensor adapter."""
    mock_adapter_dir = tmp_path / "mock_adapter"
    mock_adapter_dir.mkdir()

    # Save mock adapter config
    config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    with open(mock_adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    # Save mock tensors (only 2 tensors for layer 0)
    mock_weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, 64),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 16),
    }
    save_file(mock_weights, str(mock_adapter_dir / "adapter_model.safetensors"))

    lora_cfg = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    # Must raise ValueError highlighting incomplete layer coverage
    with pytest.raises(ValueError, match="Incomplete layer coverage"):
        validate_saved_adapter(
            adapter_dir=str(mock_adapter_dir),
            expected_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            lora_cfg=lora_cfg,
        )


def test_validator_rejects_mismatched_base_model(tmp_path):
    """Verify that validate_saved_adapter rejects an adapter configured for a different base model."""
    adapter_dir = tmp_path / "mismatched_adapter"
    adapter_dir.mkdir()

    config = {
        "base_model_name_or_path": "gpt2-small",
        "r": 16,
        "target_modules": ["q_proj"],
    }
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    save_file({"some_weight": torch.randn(10, 10)}, str(adapter_dir / "adapter_model.safetensors"))

    with pytest.raises(ValueError, match="Adapter base model mismatch"):
        validate_saved_adapter(
            adapter_dir=str(adapter_dir),
            expected_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            lora_cfg={"r": 16, "target_modules": ["q_proj"]},
        )


def test_validator_rejects_missing_target_modules(tmp_path):
    """Verify that validate_saved_adapter rejects an adapter missing configured target modules."""
    adapter_dir = tmp_path / "missing_modules_adapter"
    adapter_dir.mkdir()

    # Config declares 7 target modules
    config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    # But weights only contain q_proj for 28 layers (missing 6 other modules)
    weights = {}
    for i in range(28):
        weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"] = torch.randn(16, 1536)
        weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"] = torch.randn(1536, 16)
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))

    lora_cfg = {
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    with pytest.raises(ValueError, match="Missing target modules in safetensors weights"):
        validate_saved_adapter(
            adapter_dir=str(adapter_dir),
            expected_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            lora_cfg=lora_cfg,
        )


def test_quarantined_adapter_provenance_failure():
    """Verify that the backed-up pre-audit adapter artifact fails provenance audit."""
    quarantine_path = Path("outputs/quarantine_pre_audit/final_adapter")
    if quarantine_path.exists():
        lora_cfg = {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        with pytest.raises(ValueError, match="Incomplete layer coverage"):
            validate_saved_adapter(
                adapter_dir=str(quarantine_path),
                expected_model_id="Qwen/Qwen2.5-1.5B-Instruct",
                lora_cfg=lora_cfg,
            )
