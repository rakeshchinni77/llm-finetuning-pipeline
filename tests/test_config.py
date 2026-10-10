"""Unit tests for configuration loading and validation."""

import os
import pytest
from src.utils.config import load_yaml, merge_configs, validate_evaluation_config, validate_training_config


def test_load_valid_yaml():
    """Verify that training and evaluation configs load successfully."""
    cfg = load_yaml("configs/training_config.yaml")
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "lora" in cfg
    assert "training" in cfg


def test_validate_training_config_success():
    """Verify standard training config passes validation."""
    cfg = load_yaml("configs/training_config.yaml")
    validate_training_config(cfg)  # Should not raise


def test_validate_training_config_missing_section():
    """Verify error raised when required section is missing."""
    cfg = load_yaml("configs/training_config.yaml")
    del cfg["lora"]
    with pytest.raises(ValueError, match="Missing required configuration section"):
        validate_training_config(cfg)


def test_validate_training_config_invalid_splits():
    """Verify error raised when data splits do not sum to 1.0."""
    cfg = load_yaml("configs/training_config.yaml")
    cfg["data"]["train_split"] = 0.7
    cfg["data"]["val_split"] = 0.1
    cfg["data"]["test_split"] = 0.1  # Sum is 0.9
    with pytest.raises(ValueError, match="Data splits must sum to 1.0"):
        validate_training_config(cfg)


def test_validate_training_config_invalid_lora():
    """Verify error raised when LoRA rank is invalid."""
    cfg = load_yaml("configs/training_config.yaml")
    cfg["lora"]["r"] = -4
    with pytest.raises(ValueError, match="LoRA rank 'r' must be a positive integer"):
        validate_training_config(cfg)


def test_merge_configs():
    """Verify nested configuration overrides merge correctly."""
    base = {
        "model": {"model_id": "base-model", "dtype": "fp16"},
        "training": {"lr": 1e-4, "batch_size": 2}
    }
    overrides = {
        "model": {"model_id": "custom-model"},
        "training": {"batch_size": 4}
    }
    merged = merge_configs(base, overrides)
    assert merged["model"]["model_id"] == "custom-model"
    assert merged["model"]["dtype"] == "fp16"
    assert merged["training"]["lr"] == 1e-4
    assert merged["training"]["batch_size"] == 4


def test_validate_evaluation_config_success():
    """Verify evaluation config passes validation."""
    cfg = load_yaml("configs/evaluation_config.yaml")
    validate_evaluation_config(cfg)
