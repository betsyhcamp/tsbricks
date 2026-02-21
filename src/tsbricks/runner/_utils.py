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
        ModuleNotFoundError: If the module portion cannot be imported.
        AttributeError: If the attribute does not exist on the module.
    """
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
