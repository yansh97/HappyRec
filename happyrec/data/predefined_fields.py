from . import field_types as ftp
from .field import FieldType

FTYPES: dict[str, FieldType] = {}
"""The field types of the predefined fields."""

# Fields associated with the interactions.
UID = "uid"
"""User ID."""
FTYPES[UID] = ftp.category()

IID = "iid"
"""Item ID."""
FTYPES[IID] = ftp.category()

LABEL = "label"
"""Label."""
FTYPES[LABEL] = ftp.float_()

TIMESTAMP = "timestamp"
"""Timestamp (optional)."""
FTYPES[TIMESTAMP] = ftp.int_()

TRAIN_MASK = "train_mask"
"""Training dataset mask (after splitting)."""
FTYPES[TRAIN_MASK] = ftp.bool_()

VAL_MASK = "val_mask"
"""Validation dataset mask (after splitting)."""
FTYPES[VAL_MASK] = ftp.bool_()

TEST_MASK = "test_mask"
"""Test dataset mask (after splitting)."""
FTYPES[TEST_MASK] = ftp.bool_()

# Fields associated with the users.
TRAIN_IIDS_SET = "train_iids_set"
"""Set of item IDs interacted by the user in the training dataset (after splitting)."""
FTYPES[TRAIN_IIDS_SET] = ftp.object_()

VAL_IIDS_SET = "val_iids_set"
"""Set of item IDs interacted by the user in the validation dataset (after splitting).
"""
FTYPES[VAL_IIDS_SET] = ftp.object_()

TEST_IIDS_SET = "test_iids_set"
"""Set of item IDs interacted by the user in the test dataset (after splitting)."""
FTYPES[TEST_IIDS_SET] = ftp.object_()

VAL_NEG_IIDS = "val_neg_iids"
"""Array of negative item IDs for validation (after eval negative sampling)."""
FTYPES[VAL_NEG_IIDS] = ftp.fixed_size_list(ftp.category())

TEST_NEG_IIDS = "test_neg_iids"
"""Array of negative item IDs for test (after eval negative sampling)."""
FTYPES[TEST_NEG_IIDS] = ftp.fixed_size_list(ftp.category())

# Fields associated with the items.
POP_PROB = "pop_prob"
"""Popularity probability of items (after splitting)."""
FTYPES[POP_PROB] = ftp.float_()
