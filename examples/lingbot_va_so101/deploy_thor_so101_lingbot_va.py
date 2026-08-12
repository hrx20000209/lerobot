#!/usr/bin/env python
"""Compatibility entry point for the Thor SO-101 LingBot-VA deployment script."""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inference" / "deploy_thor_so101_lingbot_va.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
