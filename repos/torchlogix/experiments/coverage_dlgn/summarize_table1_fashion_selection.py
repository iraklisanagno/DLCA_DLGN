#!/usr/bin/env python3
"""Aggregate the locked Table 1 Fashion-MNIST selection runs."""

from pathlib import Path

import summarize_table1_selection as selection


selection.QUEUE_PHASE = "table1_select_fashion"
selection.RESULT_PREFIX = "select_table1_fashion"
selection.SUMMARY_STEM = "table1_fashion_selection"
selection.QUEUE_SUMMARY = (
    Path("experiments/coverage_dlgn/logs")
    / selection.QUEUE_PHASE
    / "queue_summary.json"
)


if __name__ == "__main__":
    selection.main()
