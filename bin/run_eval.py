#!/usr/bin/env python3
"""Repository-local CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_eval.cli import main

raise SystemExit(main())
