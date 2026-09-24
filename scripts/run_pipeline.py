"""
run_pipeline.py · NaviGuard — Full Pipeline Runner
──────────────────────────────────────────────────────
Thin wrapper around `naviguard pipeline` (generate -> preprocess -> train ->
predict). All options are forwarded, e.g.:

    python scripts/run_pipeline.py --n-satellites 3 --anomaly-count 6 --epochs 30
"""

import sys

from naviguard.cli import main

if __name__ == "__main__":
    sys.exit(main(["pipeline", *sys.argv[1:]]))
