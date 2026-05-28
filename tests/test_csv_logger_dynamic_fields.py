from __future__ import annotations

import csv
import sys
from pathlib import Path


def _csv_logger_cls():
    submodule_root = Path(__file__).resolve().parents[1]
    package_root = submodule_root / "koochak"
    sys.path.insert(0, str(submodule_root))

    import koochak

    if hasattr(koochak, "__path__") and str(package_root) not in koochak.__path__:
        koochak.__path__ = [str(package_root), *list(koochak.__path__)]

    from koochak.logging.csv import CSVLogger

    return CSVLogger


def test_csv_logger_preserves_eval_metrics_added_after_train_header(tmp_path):
    CSVLogger = _csv_logger_cls()
    path = tmp_path / "log.csv"
    logger = CSVLogger(str(path))

    logger.write({"step": 0, "loss": 1.0})
    logger.write({"step": 0, "val_loss": 2.0, "val_accuracy": 0.5})
    logger.write({"step": 1, "loss": 0.75})

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["loss"] == "1.0"
    assert rows[1]["val_loss"] == "2.0"
    assert rows[1]["val_accuracy"] == "0.5"
    assert rows[2]["loss"] == "0.75"


def test_csv_logger_reuses_existing_header_on_resume(tmp_path):
    CSVLogger = _csv_logger_cls()
    path = tmp_path / "log.csv"
    logger = CSVLogger(str(path))
    logger.write({"step": 0, "loss": 1.0})

    resumed = CSVLogger(str(path))
    resumed.write({"step": 1, "loss": 0.5, "val_loss": 0.75})

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["loss"] == "1.0"
    assert rows[1]["loss"] == "0.5"
    assert rows[1]["val_loss"] == "0.75"
