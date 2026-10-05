"""The prerecorded replay entry point uses the same offline policy/processor path."""
from .infer_offline import main

if __name__ == "__main__":
    raise SystemExit(main())
