"""Quick planning test: 3 episodes, small MPPI budget.

Example:
    python3 scripts/smoke_eval.py checkpoint=outputs/pusht_factored_seed0/model.pt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    extra = [a for a in sys.argv[1:] if not a.startswith("+fast=")]
    if not any(a.startswith("checkpoint=") for a in extra):
        print("Usage: python3 scripts/smoke_eval.py checkpoint=path/to/model.pt")
        sys.exit(1)
    cmd = [
        sys.executable,
        str(repo / "scripts" / "evaluate.py"),
        "fast=true",
        *extra,
    ]
    subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()
