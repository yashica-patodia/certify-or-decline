#!/usr/bin/env python3
"""Shared GPQA helpers used by both input preparation and analysis, so the two
sides can never silently disagree on how an option block is parsed."""

from __future__ import annotations

import re


def options_from_question(question: str) -> dict[str, str]:
    """Parse the ``A) ... D) ...`` option block out of a GPQA question."""
    matches = re.finditer(
        r"(?ms)^[ \t]*([A-D])\)[ \t]*(.*?)"
        r"(?=^[ \t]*[A-D]\)[ \t]*|\Z)",
        question,
    )
    return {match.group(1): match.group(2).strip() for match in matches}
