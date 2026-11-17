"""Evaluation metrics."""


def accuracy(y_true, y_pred) -> float:
    """Share of positions (e.g. frames) where prediction and label agree."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    if not y_true:
        raise ValueError("Empty input")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def spells(labels, valid=None) -> list[tuple[str, int, int]]:
    """Maximal runs of one label: (label, first, last) with last inclusive. valid: a mask of the
    positions that count; a run ends where it is False, and positions outside it form no spell."""
    import numpy as np

    labels = np.asarray(labels).astype(str)
    valid = np.ones(len(labels), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    runs, start = [], None
    for i in range(len(labels) + 1):
        if start is not None and (i == len(labels) or not valid[i] or labels[i] != labels[start]):
            runs.append((labels[start], start, i - 1))
            start = None
        if i < len(labels) and valid[i] and start is None:
            start = i
    return runs


def spell_f1(y_true, y_pred, min_iou: float = 0.5, labels: str = "HAD") -> dict:
    """Spell F1, our approximation of the official metric: the spells (runs of one label) of the
    prediction and the truth, over the positions the truth grades (labels). A predicted spell is
    a hit if it overlaps a true spell of the same label by intersection over union >= min_iou;
    each spell is matched once at most, the best overlaps first. Returns per label and overall
    (micro over all spells) precision, recall and F1, and macro F1 over the labels."""
    import numpy as np

    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    graded = np.isin(y_true, list(labels))
    true_spells, pred_spells = spells(y_true, graded), spells(y_pred, graded)
    result, totals = {}, np.zeros(3)  # hits, predicted, true
    for label in labels:
        truth = [(a, b) for lab, a, b in true_spells if lab == label]
        pred = [(a, b) for lab, a, b in pred_spells if lab == label]
        pairs = []
        for i, (pa, pb) in enumerate(pred):
            for j, (ta, tb) in enumerate(truth):
                overlap = min(pb, tb) - max(pa, ta) + 1
                if overlap > 0:
                    iou = overlap / (max(pb, tb) - min(pa, ta) + 1)
                    if iou >= min_iou:
                        pairs.append((iou, i, j))
        used_pred, used_true = set(), set()
        for _, i, j in sorted(pairs, reverse=True):
            if i not in used_pred and j not in used_true:
                used_pred.add(i)
                used_true.add(j)
        hits = len(used_pred)
        result[label] = _prf(hits, len(pred), len(truth))
        totals += [hits, len(pred), len(truth)]
    result["all"] = _prf(*totals)
    result["macro_f1"] = float(np.mean([result[label]["f1"] for label in labels]))
    return result


def _prf(hits, predicted, true) -> dict:
    precision = hits / predicted if predicted else 0.0
    recall = hits / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"spells_true": int(true), "spells_pred": int(predicted), "hits": int(hits),
            "precision": precision, "recall": recall, "f1": f1}
