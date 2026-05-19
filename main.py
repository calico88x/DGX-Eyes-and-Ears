"""
main.py — Eyes and Ears entry point

Usage
-----
    python main.py                      # uses ./config, sim mode per simulation.yaml
    python main.py --config ./config    # explicit config dir
    python main.py --sim                # force simulation mode
    python main.py --help
"""

import argparse
import sys

from app.utils.config_loader import load_config, ConfigError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eyes and Ears — AI conversational harness"
    )
    parser.add_argument(
        "--config", default="./config",
        help="Path to config directory (default: ./config)"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Force simulation mode regardless of simulation.yaml"
    )
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"\nConfig error:\n{exc}", file=sys.stderr)
        sys.exit(1)

    # Allow CLI flag to override config file
    if args.sim:
        config.simulation.enabled = True

    # Start pipeline
    from app.core.loop import PipelineLoop
    loop = PipelineLoop(config)
    loop.start()


if __name__ == "__main__":
    main()
