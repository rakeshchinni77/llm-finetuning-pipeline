"""Diagnostic and audit utilities for verifying adapter provenance and training logs."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from safetensors import safe_open


def get_expected_model_architecture(model_id: str) -> Dict[str, Any]:
    """Returns expected architectural specs for supported open-weight models."""
    if "Qwen2.5-1.5B" in model_id or "qwen2.5-1.5b" in model_id.lower():
        return {
            "num_layers": 28,
            "hidden_size": 1536,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "expected_tensors_per_layer": 14,  # 7 modules * 2 (lora_A, lora_B)
            "expected_total_tensors": 28 * 14,  # 392 tensors
            "min_expected_params": 10_000_000,
            "max_expected_params": 15_000_000,
        }
    elif "Phi-3-mini" in model_id:
        return {
            "num_layers": 32,
            "hidden_size": 3072,
            "target_modules": ["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
            "expected_tensors_per_layer": 8,
            "expected_total_tensors": 32 * 8,
            "min_expected_params": 8_000_000,
            "max_expected_params": 30_000_000,
        }
    elif "Mistral-7B" in model_id:
        return {
            "num_layers": 32,
            "hidden_size": 4096,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "expected_tensors_per_layer": 14,
            "expected_total_tensors": 32 * 14,
            "min_expected_params": 20_000_000,
            "max_expected_params": 50_000_000,
        }
    return {
        "num_layers": 1,
        "hidden_size": 64,
        "target_modules": [],
        "expected_tensors_per_layer": 2,
        "expected_total_tensors": 2,
        "min_expected_params": 1000,
        "max_expected_params": 100_000_000,
    }


def audit_adapter_weights(adapter_dir: str, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Dict[str, Any]:
    """Audits adapter directory to determine if it is a production model adapter or mock/synthetic."""
    adapter_path = Path(adapter_dir)
    safetensors_file = adapter_path / "adapter_model.safetensors"
    config_file = adapter_path / "adapter_config.json"

    report: Dict[str, Any] = {
        "adapter_dir": str(adapter_dir),
        "safetensors_exists": safetensors_file.is_file(),
        "config_exists": config_file.is_file(),
        "is_genuine_production_adapter": False,
        "tensor_count": 0,
        "total_parameters": 0,
        "detected_layers": [],
        "detected_modules": [],
        "verdict": "",
        "discrepancies": [],
    }

    if not report["safetensors_exists"]:
        report["verdict"] = "MISSING_SAFETENSORS: adapter_model.safetensors does not exist."
        return report

    # Load adapter_config.json if present
    adapter_cfg = {}
    if report["config_exists"]:
        with open(config_file, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        report["config_base_model"] = adapter_cfg.get("base_model_name_or_path")
        report["config_lora_r"] = adapter_cfg.get("r")
        report["config_target_modules"] = adapter_cfg.get("target_modules", [])

    # Inspect safetensors file
    with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        report["tensor_count"] = len(keys)
        total_params = sum(f.get_tensor(k).numel() for k in keys)
        report["total_parameters"] = total_params

        # Extract layer indices and module names
        layers = set()
        modules = set()
        for k in keys:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layers.add(int(parts[i + 1]))
                if any(m in p for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                    modules.add(p)

        report["detected_layers"] = sorted(list(layers))
        report["detected_modules"] = sorted(list(modules))

    # Architecture expectations
    spec = get_expected_model_architecture(model_id)
    expected_layers = spec["num_layers"]
    expected_tensors = spec["expected_total_tensors"]
    min_params = spec["min_expected_params"]

    if len(keys) < expected_tensors:
        report["discrepancies"].append(
            f"Tensor count mismatch: Found {len(keys)} tensors, but {model_id} requires {expected_tensors} tensors across {expected_layers} layers."
        )

    if len(layers) < expected_layers:
        report["discrepancies"].append(
            f"Layer coverage mismatch: Found {len(layers)} layers (layers {report['detected_layers']}), expected {expected_layers} layers (0-{expected_layers-1})."
        )

    if total_params < min_params:
        report["discrepancies"].append(
            f"Parameter count mismatch: Adapter has {total_params:,} parameters, which is far below expected ~12.9M parameters for {model_id} LoRA r=16."
        )

    if not report["discrepancies"]:
        report["is_genuine_production_adapter"] = True
        report["verdict"] = "GENUINE_PRODUCTION_ADAPTER: All layers, target modules, and parameter counts align with the base model architecture."
    else:
        report["is_genuine_production_adapter"] = False
        report["verdict"] = "PROVENANCE_FAIL: Synthetic/mock or incomplete adapter detected. " + "; ".join(report["discrepancies"])

    return report


def audit_tensorboard_provenance(logdir: str = "runs") -> Dict[str, Any]:
    """Audits TensorBoard event logs to determine if they reflect a real multi-step training loop or synthetic emission."""
    log_path = Path(logdir)
    report: Dict[str, Any] = {
        "logdir": str(logdir),
        "event_files_found": [],
        "is_genuine_live_training": False,
        "scalar_tags": [],
        "step_count": 0,
        "wall_time_duration_seconds": 0.0,
        "verdict": "",
    }

    event_files = list(log_path.glob("**/events.out.tfevents*"))
    report["event_files_found"] = [str(p) for p in event_files]

    if not event_files:
        report["verdict"] = "NO_EVENTS: No TensorBoard event files found."
        return report

    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(str(event_files[0].parent))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        report["scalar_tags"] = tags

        if "train/loss" in tags:
            events = ea.Scalars("train/loss")
            report["step_count"] = len(events)
            if len(events) >= 2:
                duration = events[-1].wall_time - events[0].wall_time
                report["wall_time_duration_seconds"] = round(duration, 4)

                # Check if all steps share identical timestamps (within 0.01 seconds)
                if duration < 0.05:
                    report["verdict"] = (
                        f"SYNTHETIC_LOG_DETECTED: {len(events)} steps logged instantaneously in {duration}s. "
                        "Indicates scripted batch emission rather than incremental step-by-step SFTTrainer progression."
                    )
                    report["is_genuine_live_training"] = False
                else:
                    report["verdict"] = f"GENUINE_TRAINING_LOG: {len(events)} steps logged across {duration:.1f}s."
                    report["is_genuine_live_training"] = True
    except Exception as e:
        report["verdict"] = f"ERROR_PARSING_LOGS: {e}"

    return report
