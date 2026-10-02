import os

# Use a non-GUI Matplotlib backend so batch runs work reliably on Windows/VS Code.
os.environ.setdefault("MPLBACKEND", "Agg")

from src.screen_yield_sensitivity import main as run_screen
from src.walk_forward_backtest import main as run_backtest


if __name__ == "__main__":
    run_screen()
    run_backtest()
