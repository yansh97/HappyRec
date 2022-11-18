from typing import Callable, TypeVar

CLS = TypeVar("CLS", bound=type)


def add_init_doc(doc: str) -> Callable[[CLS], CLS]:
    """Add a docstring for the ``__init__`` method of a class without an explicit
    ``__init__`` method.

    :param doc: The docstring.
    :return: A decorator that adds the docstring to the ``__init__`` method of the
        decorated class.
    """

    def decorator(cls: CLS) -> CLS:
        cls.__init__.__doc__ = doc
        return cls

    return decorator
