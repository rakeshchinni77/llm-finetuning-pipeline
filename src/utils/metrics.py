"""Evaluation metrics calculation module: ROUGE scores and Perplexity."""

from typing import Dict, List, Optional
import math
import torch


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True
) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for model outputs.
    
    Args:
        predictions: List of generated response strings.
        references: List of target ground-truth strings.
        use_stemmer: Whether to use Porter stemmer for word normalization.
        
    Returns:
        Dictionary mapping metric names (e.g. 'rouge1', 'rouge2', 'rougeL') to F1 scores in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions count ({len(predictions)}) does not match references count ({len(references)})"
        )
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Clean empty strings to prevent evaluation errors
    clean_preds = [p.strip() if p.strip() else "<empty>" for p in predictions]
    clean_refs = [r.strip() if r.strip() else "<empty>" for r in references]

    try:
        import evaluate
        rouge_metric = evaluate.load("rouge")
        results = rouge_metric.compute(
            predictions=clean_preds,
            references=clean_refs,
            use_stemmer=use_stemmer
        )
        return {
            "rouge1": round(float(results["rouge1"]), 4),
            "rouge2": round(float(results["rouge2"]), 4),
            "rougeL": round(float(results["rougeL"]), 4),
        }
    except Exception:
        # Fallback to direct rouge_score library if evaluate isn't cached or available
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
        scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(clean_preds, clean_refs):
            score = scorer.score(ref, pred)
            scores["rouge1"] += score["rouge1"].fmeasure
            scores["rouge2"] += score["rouge2"].fmeasure
            scores["rougeL"] += score["rougeL"].fmeasure
        n = len(clean_preds)
        return {
            "rouge1": round(scores["rouge1"] / n, 4),
            "rouge2": round(scores["rouge2"] / n, 4),
            "rougeL": round(scores["rougeL"] / n, 4),
        }


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 1024,
    device: Optional[str] = None
) -> float:
    """Computes perplexity across a list of target texts using causal cross-entropy loss.
    
    Args:
        model: Causal language model.
        tokenizer: Pre-trained tokenizer.
        texts: List of text sequences.
        max_length: Maximum sequence token length.
        device: Device to place tensors on. If None, uses model device.
        
    Returns:
        Computed perplexity as float.
    """
    if not texts:
        return float("nan")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = encodings.input_ids.to(device)
            target_ids = input_ids.clone()

            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.size(1)

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("nan")

    avg_nll = total_loss / total_tokens
    try:
        ppl = math.exp(avg_nll)
    except OverflowError:
        ppl = float("inf")
    return round(ppl, 4)
