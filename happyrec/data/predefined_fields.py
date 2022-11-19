from .field import FieldType, ItemType, ScalarType

FTYPES: dict[str, FieldType] = {}
"""The field types of the predefined fields."""

# Fields associated with the interactions.
UID = "uid"
"""User ID."""
FTYPES[UID] = FieldType(ScalarType.CATEGORICAL, ItemType.SCALAR)

IID = "iid"
"""Item ID."""
FTYPES[IID] = FieldType(ScalarType.CATEGORICAL, ItemType.SCALAR)

LABEL = "label"
"""Label."""
FTYPES[LABEL] = FieldType(ScalarType.FLOAT, ItemType.SCALAR)

TIMESTAMP = "timestamp"
"""Timestamp (optional)."""
FTYPES[TIMESTAMP] = FieldType(ScalarType.INT, ItemType.SCALAR)

TRAIN_MASK = "train_mask"
"""Training dataset mask (after splitting)."""
FTYPES[TRAIN_MASK] = FieldType(ScalarType.BOOL, ItemType.SCALAR)

VAL_MASK = "val_mask"
"""Validation dataset mask (after splitting)."""
FTYPES[VAL_MASK] = FieldType(ScalarType.BOOL, ItemType.SCALAR)

TEST_MASK = "test_mask"
"""Test dataset mask (after splitting)."""
FTYPES[TEST_MASK] = FieldType(ScalarType.BOOL, ItemType.SCALAR)

# Fields associated with the users.
ORIGINAL_UID = "original_uid"
"""Original user ID."""
FTYPES[ORIGINAL_UID] = FieldType(ScalarType.OBJECT, ItemType.SCALAR)

TRAIN_IIDS_SET = "train_iids_set"
"""Set of item IDs interacted by the user in the training dataset (after splitting)."""
FTYPES[TRAIN_IIDS_SET] = FieldType(ScalarType.OBJECT, ItemType.SCALAR)

VAL_IIDS_SET = "val_iids_set"
"""Set of item IDs interacted by the user in the validation dataset (after splitting).
"""
FTYPES[VAL_IIDS_SET] = FieldType(ScalarType.OBJECT, ItemType.SCALAR)

TEST_IIDS_SET = "test_iids_set"
"""Set of item IDs interacted by the user in the test dataset (after splitting)."""
FTYPES[TEST_IIDS_SET] = FieldType(ScalarType.OBJECT, ItemType.SCALAR)

VAL_NEG_IIDS = "val_neg_iids"
"""Array of negative item IDs for validation (after eval negative sampling)."""
FTYPES[VAL_NEG_IIDS] = FieldType(ScalarType.CATEGORICAL, ItemType.ARRAY)

TEST_NEG_IIDS = "test_neg_iids"
"""Array of negative item IDs for test (after eval negative sampling)."""
FTYPES[TEST_NEG_IIDS] = FieldType(ScalarType.CATEGORICAL, ItemType.ARRAY)

# Fields associated with the items.
ORIGINAL_IID = "original_iid"
"""Original item ID."""
FTYPES[ORIGINAL_IID] = FieldType(ScalarType.OBJECT, ItemType.SCALAR)

POP_PROB = "pop_prob"
"""Popularity probability of items (after splitting)."""
FTYPES[POP_PROB] = FieldType(ScalarType.FLOAT, ItemType.SCALAR)
