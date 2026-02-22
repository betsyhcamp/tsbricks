from __future__ import annotations

import importlib
from typing import Any


def dynamic_import(dotted_path: str) -> Any:
    """Import and return an attribute from a dotted module path.

    For example, ``dynamic_import("tsbricks.blocks.transforms.BoxCoxTransform")``
    imports ``tsbricks.blocks.transforms`` and returns the ``BoxCoxTransform``
    attribute.

    Args:
        dotted_path: Fully-qualified ``module.attribute`` string.

    Raises:
        ValueError: If the ``dotted_path`` does not have dot in
            expected `module.attribute` format
        ModuleNotFoundError: If the module portion cannot be imported.
        AttributeError: If the attribute does not exist on the module.
    """
    if "." not in dotted_path:
        raise ValueError(
            f"Invalid dotted path {dotted_path!r}: expected 'module.attribute' format"
        )
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
