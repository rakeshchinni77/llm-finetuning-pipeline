"""Unit tests for dataset acquisition, ChatML formatting, and non-overlapping splits."""

import pytest
from src.data_pipeline import (
    clean_and_format_dataset,
    format_chatml,
    generate_curated_instruction_dataset,
    split_dataset,
)


def test_format_chatml():
    """Verify ChatML format structure and prompt/response separation."""
    sample = format_chatml(
        instruction="What is QLoRA?",
        context="Domain: Deep Learning",
        response="QLoRA is Quantized Low-Rank Adaptation."
    )
    assert "instruction" in sample
    assert "context" in sample
    assert "response" in sample
    assert "prompt" in sample
    assert "completion" in sample
    assert "text" in sample

    # Check delimiters and schema integrity
    assert "<|im_start|>system" in sample["prompt"]
    assert "<|im_start|>user" in sample["prompt"]
    assert "<|im_start|>assistant\n" in sample["prompt"]
    assert sample["response"] not in sample["prompt"], "Response must NOT leak into input prompt!"
    assert sample["completion"] == f"{sample['response']}<|im_end|>"
    assert sample["text"] == sample["prompt"] + sample["completion"]
    assert sample["text"].endswith("<|im_end|>")


def test_generate_curated_dataset_size():
    """Verify curated dataset generation satisfies >= 1,000 examples requirement."""
    dataset = generate_curated_instruction_dataset(num_samples=1200, seed=42)
    assert len(dataset) == 1200
    for item in dataset[:10]:
        assert len(item["instruction"]) > 10
        assert len(item["response"]) > 20


def test_clean_and_format_deduplication():
    """Verify deduplication and cleaning logic."""
    raw = [
        {"instruction": "Explain LoRA", "context": "", "response": "LoRA uses low rank matrices."},
        {"instruction": "Explain LoRA", "context": "", "response": "LoRA uses low rank matrices."},  # Duplicate
        {"instruction": "Hi", "context": "", "response": "ok"},  # Too short
    ]
    cleaned = clean_and_format_dataset(raw)
    assert len(cleaned) == 1
    assert cleaned[0]["instruction"] == "Explain LoRA"


def test_split_dataset_proportions_and_no_leakage():
    """Verify exact 80/10/10 split and guarantee zero overlap between subsets."""
    dataset = generate_curated_instruction_dataset(num_samples=1000, seed=42)
    formatted = clean_and_format_dataset(dataset)

    train_data, val_data, test_data = split_dataset(
        formatted,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    n_total = len(formatted)
    assert len(train_data) == int(n_total * 0.8)
    assert len(val_data) == int(n_total * 0.1)
    assert len(test_data) == n_total - len(train_data) - len(val_data)

    # Verify zero data leakage
    train_prompts = set(x["prompt"] for x in train_data)
    val_prompts = set(x["prompt"] for x in val_data)
    test_prompts = set(x["prompt"] for x in test_data)

    assert len(train_prompts.intersection(val_prompts)) == 0
    assert len(train_prompts.intersection(test_prompts)) == 0
    assert len(val_prompts.intersection(test_prompts)) == 0


def test_sft_trainer_dataset_schema_regression():
    """Regression test: verify SFTTrainer tokenization accepts the processed dataset without KeyError: 'completion'."""
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    record = format_chatml(
        instruction="What is SFT?",
        context="NLP Domain",
        response="Supervised Fine-Tuning trains an LLM on instruction-response pairs."
    )
    dataset = Dataset.from_dict({k: [v] for k, v in record.items()})

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    cfg_m = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    cfg_m.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(cfg_m)

    cfg = SFTConfig(output_dir="outputs/test_schema", max_length=64, use_cpu=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok)

    prepared = trainer.train_dataset[0]
    assert "input_ids" in prepared
    assert "labels" in prepared
    assert len(prepared["input_ids"]) > 0
    # Confirm prompt tokens are masked in labels with -100
    assert -100 in prepared["labels"]

