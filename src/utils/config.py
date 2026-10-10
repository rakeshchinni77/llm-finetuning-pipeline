"""Configuration management and validation utilities for fine-tuning pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load local environment variables from .env file if available
load_dotenv()


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Reads and parses a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If parsing fails or the content is empty.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration in {file_path} must be a valid key-value mapping.")

    return config


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary.
    
    Args:
        base: The base configuration dictionary.
        overrides: The overrides to apply.
        
    Returns:
        Updated dictionary.
    """
    result = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validates required sections and keys in a training configuration.
    
    Args:
        config: Loaded training configuration dictionary.
        
    Raises:
        ValueError: If required sections or parameters are missing or invalid.
    """
    required_sections = ["model", "quantization", "lora", "data", "training"]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing required configuration section: '{section}'")

    # Validate model parameters
    if "model_id" not in config["model"] or not config["model"]["model_id"]:
        raise ValueError("Configuration 'model.model_id' must be specified.")

    # Validate LoRA parameters
    lora = config["lora"]
    for param in ["r", "lora_alpha", "target_modules"]:
        if param not in lora:
            raise ValueError(f"Missing required LoRA parameter: 'lora.{param}'")
    if not isinstance(lora["r"], int) or lora["r"] <= 0:
        raise ValueError(f"LoRA rank 'r' must be a positive integer, got {lora['r']}")
    if not isinstance(lora["lora_alpha"], (int, float)) or lora["lora_alpha"] <= 0:
        raise ValueError(f"LoRA 'lora_alpha' must be positive, got {lora['lora_alpha']}")
    if not isinstance(lora["target_modules"], list) or len(lora["target_modules"]) == 0:
        raise ValueError("LoRA 'target_modules' must be a non-empty list of module names.")

    # Validate data split ratios
    data = config["data"]
    train_split = data.get("train_split", 0.8)
    val_split = data.get("val_split", 0.1)
    test_split = data.get("test_split", 0.1)
    total_split = round(train_split + val_split + test_split, 4)
    if total_split != 1.0:
        raise ValueError(f"Data splits must sum to 1.0 (train: {train_split}, val: {val_split}, test: {test_split})")

    # Validate training hyperparameters
    training = config["training"]
    if training.get("learning_rate", 0) <= 0:
        raise ValueError(f"Invalid learning rate: {training.get('learning_rate')}")
    if training.get("per_device_train_batch_size", 0) <= 0:
        raise ValueError(f"Invalid train batch size: {training.get('per_device_train_batch_size')}")


def validate_evaluation_config(config: Dict[str, Any]) -> None:
    """Validates required sections and keys in an evaluation configuration.
    
    Args:
        config: Loaded evaluation configuration dictionary.
        
    Raises:
        ValueError: If required sections or parameters are missing.
    """
    required_sections = ["model", "data", "generation", "metrics", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required evaluation section: '{section}'")

    if not config["model"].get("base_model_id"):
        raise ValueError("Missing 'model.base_model_id' in evaluation configuration.")
    if not config["data"].get("test_data_path"):
        raise ValueError("Missing 'data.test_data_path' in evaluation configuration.")
