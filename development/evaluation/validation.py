"""Validation of predictions against ground truth."""
from evaluation.metrics import accuracy


def evaluate(predictions, ground_truth) -> dict:
    return {"accuracy": accuracy(ground_truth, predictions)}
