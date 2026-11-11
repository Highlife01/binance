"""Kesintisiz arka plan botu başlatıcısı. Ayrıntılar: src/binance_ai_bot/runner.py"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_ai_bot.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
