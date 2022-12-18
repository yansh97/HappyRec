from dataclasses import dataclass

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class ElementType:
    def __str__(self) -> str:
        raise NotImplementedError

    def _valid_dtypes(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _valid_pa_types(self) -> tuple[pa.DataType, ...]:
        raise NotImplementedError

    def _convert_dtype_to_pa_type(self, dtype: str) -> pa.DataType:
        convert_dict = dict(zip(self._valid_dtypes(), self._valid_pa_types()))
        return convert_dict[dtype]


@dataclass(frozen=True, slots=True)
class BoolEtype(ElementType):
    def __str__(self) -> str:
        return "bool"

    def _valid_dtypes(self) -> tuple[str, ...]:
        return ("bool",)

    def _valid_pa_types(self) -> tuple[pa.DataType, ...]:
        return (pa.bool_(),)


@dataclass(frozen=True, slots=True)
class IntEtype(ElementType):
    def __str__(self) -> str:
        return "int"

    def _valid_dtypes(self) -> tuple[str, ...]:
        return ("int8", "int16", "int32", "int64")

    def _valid_pa_types(self) -> tuple[pa.DataType, ...]:
        return (pa.int8(), pa.int16(), pa.int32(), pa.int64())


@dataclass(frozen=True, slots=True)
class FloatEtype(ElementType):
    def __str__(self) -> str:
        return "float"

    def _valid_dtypes(self) -> tuple[str, ...]:
        return ("float16", "float32", "float64")

    def _valid_pa_types(self) -> tuple[pa.DataType, ...]:
        return (pa.float16(), pa.float32(), pa.float64())


@dataclass(frozen=True, slots=True)
class CategoryEtype(ElementType):
    def __str__(self) -> str:
        return "category"

    def _valid_dtypes(self) -> tuple[str, ...]:
        return ("uint8", "uint16", "uint32", "uint64")

    def _valid_pa_types(self) -> tuple[pa.DataType, ...]:
        return (pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64())
