from .field import (
    CategoricalType,
    FieldType,
    ItemType,
    NumericType,
    ObjectType,
    ScalarType,
)

FTYPES: dict[str, FieldType] = {}
"""The field types of the predefined fields."""

# Fields associated with the interactions.
UID = "uid"
"""User ID."""
FTYPES[UID] = CategoricalType(ItemType.SCALAR)

IID = "iid"
"""Item ID."""
FTYPES[IID] = CategoricalType(ItemType.SCALAR)

LABEL = "label"
"""Label."""
FTYPES[LABEL] = NumericType(ItemType.SCALAR, ScalarType.FLOAT)

TIMESTAMP = "timestamp"
"""Timestamp (optional)."""
FTYPES[TIMESTAMP] = NumericType(ItemType.SCALAR, ScalarType.INT)

TRAIN_MASK = "train_mask"
"""Training dataset mask (after splitting)."""
FTYPES[TRAIN_MASK] = NumericType(ItemType.SCALAR, ScalarType.BOOL)

VAL_MASK = "val_mask"
"""Validation dataset mask (after splitting)."""
FTYPES[VAL_MASK] = NumericType(ItemType.SCALAR, ScalarType.BOOL)

TEST_MASK = "test_mask"
"""Test dataset mask (after splitting)."""
FTYPES[TEST_MASK] = NumericType(ItemType.SCALAR, ScalarType.BOOL)

# Fields associated with the users.
TRAIN_IIDS_SET = "train_iids_set"
"""Set of item IDs interacted by the user in the training dataset (after splitting)."""
FTYPES[TRAIN_IIDS_SET] = ObjectType()

VAL_IIDS_SET = "val_iids_set"
"""Set of item IDs interacted by the user in the validation dataset (after splitting).
"""
FTYPES[VAL_IIDS_SET] = ObjectType()

TEST_IIDS_SET = "test_iids_set"
"""Set of item IDs interacted by the user in the test dataset (after splitting)."""
FTYPES[TEST_IIDS_SET] = ObjectType()

VAL_NEG_IIDS = "val_neg_iids"
"""Array of negative item IDs for validation (after eval negative sampling)."""
FTYPES[VAL_NEG_IIDS] = CategoricalType(ItemType.ARRAY)

TEST_NEG_IIDS = "test_neg_iids"
"""Array of negative item IDs for test (after eval negative sampling)."""
FTYPES[TEST_NEG_IIDS] = CategoricalType(ItemType.ARRAY)

# Fields associated with the items.
POP_PROB = "pop_prob"
"""Popularity probability of items (after splitting)."""
FTYPES[POP_PROB] = NumericType(ItemType.SCALAR, ScalarType.FLOAT)
