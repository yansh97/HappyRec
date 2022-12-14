from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any, overload

from jsonargparse import Namespace


def instantiate_class(
    class_type: type,
    init_args: Namespace | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate a class with the given keyword arguments.

    :param class_type: The type of the class to instantiate.
    :param init_args: The keyword arguments to pass to the class's __init__ method.
        Default: ``None``.
    :param extra_kwargs: Extra keyword arguments to pass to the class's __init__ method.
        Default: ``None``.
    :return: The instantiated class.
    """
    kwargs: dict[str, Any] = init_args.as_dict() if init_args is not None else {}
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return class_type(**kwargs)


def instantiate_config(
    class_config: Namespace, extra_kwargs: Mapping[str, Any] | None = None
) -> Any:
    """Instantiate a class configuration.

    :param class_config: The class configuration to instantiate, which must have a
        "class_path" and "init_args" key.
    :param extra_kwargs: Extra keyword arguments to pass to the class's __init__ method.
        Default: ``None``.
    :return: The instantiated subclass.
    """
    kwargs: dict[str, Any] = class_config.as_dict().get("init_args", {})
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    class_module, class_name = class_config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    class_type = getattr(module, class_name)
    return class_type(**kwargs)


@overload
def instantiate_container(
    container: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    pass


@overload
def instantiate_container(container: MutableSequence[Any]) -> MutableSequence[Any]:
    pass


@overload
def instantiate_container(container: Namespace) -> Namespace:
    pass


def instantiate_container(container):
    """Instantiate all the class configurations in the given container recursively.

    :param container: The container to instantiate, which can be a mutable mapping, a
        mutable sequence, or a Namespace.
    :raises TypeError: If the container is not a mutable mapping, a mutable sequence, or
        a Namespace.
    :return: The instantiated container.
    """

    def _instantiate_value(value):
        # If the value is a container, instantiate it recursively.
        if isinstance(value, MutableMapping | MutableSequence | Namespace):
            value = instantiate_container(value)
        # If the value is a class configuration, instantiate it after instantiating its
        # "init_args" value.
        if isinstance(value, Namespace) and "class_path" in value:
            value = instantiate_config(value)
        return value

    if isinstance(container, MutableMapping | Namespace):
        for key, value in container.items():
            container[key] = _instantiate_value(value)
        return container
    if isinstance(container, MutableSequence):
        for index, value in enumerate(container):
            container[index] = _instantiate_value(value)
        return container
    raise TypeError(f"Cannot instantiate container of type {type(container)}")
