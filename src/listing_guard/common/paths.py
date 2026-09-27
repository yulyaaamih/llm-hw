import os
from pathlib import Path

ROOT = Path(os.environ.get("LISTING_GUARD_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
