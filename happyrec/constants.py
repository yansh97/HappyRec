import os
from pathlib import Path

# Framework
FRAMEWORK_NAME = "HappyRec"
"""The name of the framework."""

# Cache directory
_DEFAULT_CACHE_DIR = f"~/.cache/{FRAMEWORK_NAME}"
CACHE_DIR = os.getenv("HAPPYREC_CACHE_DIR", _DEFAULT_CACHE_DIR)
"""The cache directory, which is ``~/.cache/HappyRec`` by default."""
_CACHE_PATH = Path(CACHE_DIR).expanduser().resolve()
_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Separator
FIELD_SEP = ","
"""The separator used to separate fields in a line."""
SEQ_SEP = " "
"""The separator used to separate items in a sequence field."""
