from src.progress import Progress, Steps, duration


def test_steps_are_numbered_and_timed(capsys):
    steps = Steps(2)
    with steps("Loading"):
        pass
    with steps("Matching"):
        pass
    err = capsys.readouterr().err
    assert "[1/2] Loading ..." in err and "[2/2] Matching ..." in err
    assert err.count("done in") == 2


def test_steps_can_be_silent(capsys):
    with Steps(1, enabled=False)("Loading"):
        pass
    assert capsys.readouterr().err == ""


def test_progress_outside_a_terminal_writes_a_line_per_ten_percent(capsys):
    with Progress(20, "matching") as bar:
        for _ in range(20):
            bar.update()
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 11  # 0 %, 10 %, ..., 100 %
    assert lines[0].strip().startswith("matching: 0/20 (0%)")
    assert lines[-1].strip().startswith("matching: 20/20 (100%)")


def test_duration():
    assert duration(12.34) == "12.3 s"
    assert duration(227) == "3:47 min"
