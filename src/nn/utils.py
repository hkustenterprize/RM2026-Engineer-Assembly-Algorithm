"""Small configuration-driven object factory shared by training tools."""

import importlib
from collections.abc import Mapping
from typing import Any


def import_and_get(object_path: str) -> Any:
    """Import and return an object addressed by its fully qualified name."""
    if "." not in object_path:
        raise ValueError(f"Expected a fully qualified object path, got: {object_path!r}")
    module_name, object_name = object_path.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"{object_path!r} does not exist") from exc


def make_object(class_path: str, args: Mapping[str, Any] | None = None) -> Any:
    """Instantiate a class using keyword arguments."""
    cls = import_and_get(class_path)
    return cls() if args is None else cls(**args)


def make_object_from_config(config: Any) -> Any:
    """Recursively instantiate `_class_name` entries in a config tree."""
    if isinstance(config, list):
        return [make_object_from_config(item) for item in config]
    if isinstance(config, dict):
        if "_function_name" in config:
            if len(config) != 1:
                raise ValueError("_function_name cannot be combined with other keys")
            return import_and_get(config["_function_name"])
        if "_class_name" in config:
            return make_object(
                config["_class_name"],
                {
                    key: make_object_from_config(value)
                    for key, value in config.items()
                    if key != "_class_name"
                },
            )
        return {key: make_object_from_config(value) for key, value in config.items()}
    return config
