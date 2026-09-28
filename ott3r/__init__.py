import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PI3_PATH = _ROOT / "external" / "pi3"

if str(_PI3_PATH) not in sys.path:
    sys.path.insert(0, str(_PI3_PATH))
