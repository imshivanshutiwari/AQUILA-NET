"""Main entry point for the AQUILA-NET training pipeline.

Usage:
    python -m training.train_aquila [--phase {a|b|c|all}]
"""

import sys
import threading
import time
import webbrowser


def _parse_phase(argv: list) -> str:
    if "--phase" in argv:
        idx = argv.index("--phase")
        return argv[idx + 1] if idx + 1 < len(argv) else "all"
    return "all"


def _open_browser(url: str, delay: float = 2.0) -> None:
    time.sleep(delay)
    webbrowser.open(url)


if __name__ == "__main__":
    from utils.config_loader import ConfigLoader
    from dashboard.app import create_app

    phase = _parse_phase(sys.argv)

    config = ConfigLoader().load_all()

    app = create_app(config=config, training_mode=True, phase=phase)

    browser_thread = threading.Thread(
        target=_open_browser,
        args=("http://localhost:8050/training",),
        daemon=True,
    )
    browser_thread.start()

    app.run(debug=False, host="0.0.0.0", port=8050)
