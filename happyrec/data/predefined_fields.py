# Fields associated with the interactions.
UID = "uid"
"""User ID."""

IID = "iid"
"""Item ID."""

LABEL = "label"
"""Label."""

TIMESTAMP = "timestamp"
"""Timestamp (optional)."""

TRAIN_MASK = "train_mask"
"""Training dataset mask (after splitting)."""

VAL_MASK = "val_mask"
"""Validation dataset mask (after splitting)."""

TEST_MASK = "test_mask"
"""Test dataset mask (after splitting)."""

# Fields associated with the users.
VAL_NEG_IIDS = "val_neg_iids"
"""Array of negative item IDs for validation (after eval negative sampling)."""

TEST_NEG_IIDS = "test_neg_iids"
"""Array of negative item IDs for test (after eval negative sampling)."""

# Fields associated with the items.
