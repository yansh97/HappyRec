import sys
from typing import TypeGuard, TypeVar

if sys.version_info >= (3, 11):
    from typing import Never
else:
    from typing_extensions import Never


IT = TypeVar("IT")
KT = TypeVar("KT")
VT = TypeVar("VT")


def assert_type(value, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise TypeError


def assert_never_type(value) -> Never:
    raise TypeError


def is_typed_list(value, item_type: type[IT]) -> TypeGuard[list[IT]]:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, item_type) for item in value)


def is_typed_dict(
    value, key_type: type[KT], value_type: type[VT]
) -> TypeGuard[dict[KT, VT]]:
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(key, key_type) and isinstance(value, value_type)
        for key, value in value.items()
    )
