"""Unit tests for ROUGE score and perplexity calculations."""

import pytest
import torch
import torch.nn as nn
from src.utils.metrics import compute_perplexity, compute_rouge_scores


def test_rouge_identical_strings():
    """Verify that identical target and prediction yield perfect ROUGE score of 1.0."""
    preds = ["Low-rank adaptation freezes base weights and trains low-rank matrices."]
    refs = ["Low-rank adaptation freezes base weights and trains low-rank matrices."]
    scores = compute_rouge_scores(preds, refs)
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_partial_overlap():
    """Verify ROUGE score on partial overlap."""
    preds = ["QLoRA uses 4-bit NormalFloat precision."]
    refs = ["LoRA uses 16-bit float precision."]
    scores = compute_rouge_scores(preds, refs)
    assert 0.0 < scores["rouge1"] < 1.0
    assert "rougeL" in scores


def test_rouge_empty_and_mismatch_handling():
    """Verify error on unequal list lengths and handling of empty inputs."""
    with pytest.raises(ValueError, match="Predictions count.*does not match"):
        compute_rouge_scores(["a"], ["b", "c"])

    empty_scores = compute_rouge_scores([], [])
    assert empty_scores["rouge1"] == 0.0


class DummyLM(nn.Module):
    """Small deterministic language model for perplexity testing."""
    def __init__(self, vocab_size=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.head = nn.Linear(32, vocab_size)

    def forward(self, input_ids, labels=None):
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return type("Output", (), {"loss": loss, "logits": logits})()


class DummyTokenizer:
    """Mock tokenizer returning token IDs."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=1024, return_tensors="pt"):
        # Map words to arbitrary dummy integer IDs
        tokens = [min(abs(hash(w)) % 90 + 2, 99) for w in text.split()]
        if not tokens:
            tokens = [2]
        return type("Encoding", (), {"input_ids": torch.tensor([tokens])})()


def test_compute_perplexity():
    """Verify perplexity computation yields a valid positive float."""
    model = DummyLM()
    tokenizer = DummyTokenizer()
    texts = [
        "Parameter efficient fine-tuning reduces VRAM requirements significantly.",
        "Supervised fine-tuning adapts pre-trained language models to instructions."
    ]
    ppl = compute_perplexity(model, tokenizer, texts, device="cpu")
    assert isinstance(ppl, float)
    assert ppl > 1.0
    assert not torch.isnan(torch.tensor(ppl))
