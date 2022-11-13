import os
from pathlib import Path

FRAMEWORK_NAME = "HappyRec"
"""The name of the framework."""

# Cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / FRAMEWORK_NAME
CACHE_DIR = Path(os.getenv("HAPPYREC_CACHE_DIR", _DEFAULT_CACHE_DIR))
"""The cache directory."""
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Separator
FIELD_SEP = ","
"""The separator used to separate fields in a line."""
SEQ_SEP = " "
"""The separator used to separate items in a sequence field."""

# Column names in the interaction data
UID = "uid:c"
"""The column name of user IDs."""
IID = "iid:c"
"""The column name of item IDs."""
LABEL = "label:n"
"""The column name of labels."""
TIME = "time:n"
"""The column name of timestamps (optional)."""
TRAIN_MASK = "train_mask:b"
"""The column name of train masks (optional)."""
VAL_MASK = "val_mask:b"
"""The column name of validation masks (optional)."""
TEST_MASK = "test_mask:b"
"""The column name of test masks (optional)."""

# Column names in the user data
ORI_UID = "ori_uid:o"
"""The column name of original user IDs (optional)."""
TRAIN_IIDS_SET = "train_iids:o"
"""The column name of item IDs which the user interacted in the training set
(optional)."""
VAL_IIDS_SET = "val_iids:o"
"""The column name of item IDs which the user interacted in the validation set
(optional)."""
TEST_IIDS_SET = "test_iids:o"
"""The column name of item IDs which the user interacted in the test set (optional)."""
VAL_NEG_IIDS = "val_neg_iids:ca"
"""The column name of negative item IDs for validation (optional)."""
TEST_NEG_IIDS = "test_neg_iids:ca"
"""The column name of negative item IDs for test (optional)."""

# Column names in the item data
ORI_IID = "ori_iid:o"
"""The column name of original item IDs (optional)."""
POP_PROB = "pop_prob:n"
"""The column name of item popularity probabilities (optional)."""

# Configurations
CONFIG_YAML = "config.yaml"
"""The name of the config YAML file."""
