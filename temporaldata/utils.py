from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple, Union, Callable, Optional, Type

import numpy as np


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    else:
        out = str(value)
    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


def serialize(
    elem,
    serialize_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    r"""
    General serialization function that handles object types that are not supported
    by hdf5. The function also opens function registry to deal with specific element
    types through `serialize_fn_map`. This function will automatically be applied to
    elements in a nested sequence structure.

    Args:
        elem: a single element to be serialized.
        serialize_fn_map: Optional dictionary mapping from element type to the
            corresponding serialize function. If the element type isn't present in this
            dictionary, it will be skipped and the element will be returned as is.
    """
    elem_type = type(elem)

    if serialize_fn_map is not None:
        if elem_type in serialize_fn_map:
            return serialize_fn_map[elem_type](elem, serialize_fn_map=serialize_fn_map)

        for object_type in serialize_fn_map:
            if isinstance(elem, object_type):
                return serialize_fn_map[object_type](
                    elem, serialize_fn_map=serialize_fn_map
                )

    if isinstance(elem, (list, tuple)):
        return elem_type(
            [serialize(e, serialize_fn_map=serialize_fn_map) for e in elem]
        )

    # element does not need to be seralized, or type not supported
    return elem
