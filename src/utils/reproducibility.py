"""Reproducibility utilities for deterministic training and hardware detection."""

import os
import random
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch to ensure reproducibility.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_hardware_info() -> Dict[str, Any]:
    """Inspects and returns current compute device and system hardware capabilities.
    
    Returns:
        Dictionary containing CUDA availability, device name, count, and memory info.
    """
    cuda_available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "current_device": None,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
    }

    if cuda_available:
        current_dev = torch.cuda.current_device()
        info["current_device"] = current_dev
        info["device_name"] = torch.cuda.get_device_name(current_dev)
        total_mem = torch.cuda.get_device_properties(current_dev).total_memory
        info["total_memory_gb"] = round(total_mem / (1024 ** 3), 2)

    return info
