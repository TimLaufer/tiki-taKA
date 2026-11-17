import pytest

from evaluation.metrics import accuracy
from evaluation.validation import evaluate


def test_accuracy():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_evaluate():
    assert evaluate(predictions=[1, 1], ground_truth=[1, 0]) == {"accuracy": 0.5}
