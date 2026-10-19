from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_TICKER = "AAPL"
DEFAULT_TIMEFRAME = "1d"
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.001