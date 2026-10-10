import logging

from visual_synthesis.progress import Progress


def test_progress_counts_items_and_logs_when_there_is_no_terminal(caplog):
    """Under pytest stderr is not a tty, so tqdm is off and periodic log lines take over."""
    with caplog.at_level(logging.INFO, logger="visual_synthesis.progress"):
        with Progress(10, "generate", "img", log_every=4) as progress:
            for _ in range(3):
                progress.advance(1)
            progress.advance(1, failed=1)  # the 4th item crosses log_every
            for _ in range(6):
                progress.advance(1)

    assert progress.done == 10
    assert progress.fields == {"failed": 1}
    lines = [record.getMessage() for record in caplog.records]
    assert any("generate 4/10" in line and "failed=1" in line for line in lines)
    assert any("generate 8/10" in line for line in lines)
