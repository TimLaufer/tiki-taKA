import numpy as np
import pytest

from evaluation.metrics import spell_f1, spells


def test_spells():
    assert spells(list("HHAAD")) == [("H", 0, 1), ("A", 2, 3), ("D", 4, 4)]
    assert spells(list("HHHH"), [True, True, False, True]) == [("H", 0, 1), ("H", 3, 3)]


def test_spell_f1():
    truth = list("HHHHHHAAAADD") + [""] * 3
    perfect = spell_f1(truth, list("HHHHHHAAAADDHHH"))
    assert perfect["all"]["f1"] == 1.0 and perfect["all"]["spells_true"] == 3  # the ungraded tail counts not
    # a short false A inside H splits it: H 0-2 (IoU 0.5, hit) and 4-5 (0.33, miss), the A 3-3 a miss
    split = spell_f1(truth, list("HHHAHHAAAADDHHH"))
    assert split["H"]["hits"] == 1 and split["H"]["spells_pred"] == 2
    assert split["all"]["precision"] == pytest.approx(3 / 5) and split["all"]["recall"] == 1.0


def test_one_match_each():
    truth = list("AAAAAAAAAA")
    assert spell_f1(truth, list("AAAAAAAAAA"), labels="A")["A"]["hits"] == 1
    split = spell_f1(np.array(truth), np.array(list("AAAAADAAAA")), labels="A")["A"]
    assert split["hits"] == 1 and split["spells_pred"] == 2  # 0-4 matches (IoU 0.5), 6-9 may not match it again
