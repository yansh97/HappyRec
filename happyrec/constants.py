import os
from pathlib import Path

# Framework
FRAMEWORK_NAME = "HappyRec"
"""The name of the framework."""

# Cache directory
DEFAULT_CACHE_DIR = f"~/.cache/{FRAMEWORK_NAME}"
"""Default cache directory."""

_CACHE_PATH = (
    Path(os.getenv("HAPPYREC_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser().resolve()
)
_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Separator
FIELD_SEP = ","
"""The separator used to separate fields in a line."""
SEQ_SEP = " "
"""The separator used to separate items in a sequence field."""

# Default seed
DEFAULT_SEED = 2022
