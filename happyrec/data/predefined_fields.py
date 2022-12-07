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
TRAIN_IIDS_SET = "train_iids_set"
"""Set of item IDs interacted by the user in the training dataset (after splitting)."""

VAL_IIDS_SET = "val_iids_set"
"""Set of item IDs interacted by the user in the validation dataset (after splitting).
"""

TEST_IIDS_SET = "test_iids_set"
"""Set of item IDs interacted by the user in the test dataset (after splitting)."""

VAL_NEG_IIDS = "val_neg_iids"
"""Array of negative item IDs for validation (after eval negative sampling)."""

TEST_NEG_IIDS = "test_neg_iids"
"""Array of negative item IDs for test (after eval negative sampling)."""

# Fields associated with the items.
POP_PROB = "pop_prob"
"""Popularity probability of items (after splitting)."""
