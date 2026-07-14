"""Compatibility entry for the reusable real SCRC harness (add default)."""

from scrc_live_e2e import main


if __name__ == "__main__":
    raise SystemExit(main("add"))

