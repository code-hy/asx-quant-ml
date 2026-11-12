"""Allow ``python -m asx_quant`` execution."""

from asx_quant.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
