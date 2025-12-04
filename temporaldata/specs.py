"""
Pydantic models for representing specifications of temporaldata objects.

These specs describe the structure, types, and shapes of Data objects and their
nested components. Specs can be generated from existing Data objects using the
`generate_spec()` method, and can be exported to dictionaries or JSON.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class AttributeSpec(BaseModel):
    """Specification for a single array attribute.

    Attributes:
        dtype: The numpy dtype as a string (e.g., 'float64', 'int64')
        shape: The shape of the array as a list of integers
    """

    model_config = ConfigDict(extra="forbid")

    dtype: str
    shape: List[int]


class ScalarSpec(BaseModel):
    """Specification for a scalar value.

    Attributes:
        type: The Python type name (e.g., 'str', 'int', 'float')
        value: The actual value (for small scalars like strings, numbers)
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    value: Optional[Any] = None


class ArrayDictSpec(BaseModel):
    """Specification for an ArrayDict object.

    ArrayDict is a dictionary of arrays that share the same first dimension.

    Attributes:
        type: Always "ArrayDict"
        length: The first dimension shared by all arrays
        attributes: Dictionary mapping attribute names to their specs
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "ArrayDict"
    length: int
    attributes: Dict[str, AttributeSpec]


class IntervalSpec(BaseModel):
    """Specification for an Interval object.

    Interval represents a set of time intervals, each defined by start and end times.

    Attributes:
        type: Always "Interval"
        length: Number of intervals
        timekeys: List of attribute names that are time-based
        attributes: Dictionary mapping attribute names to their specs
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "Interval"
    length: int
    timekeys: List[str]
    attributes: Dict[str, AttributeSpec]


class IrregularTimeSeriesSpec(BaseModel):
    """Specification for an IrregularTimeSeries object.

    IrregularTimeSeries represents event-based or irregularly sampled time series data.

    Attributes:
        type: Always "IrregularTimeSeries"
        length: Number of timestamps/events
        domain_start: Start time of the domain
        domain_end: End time of the domain
        timekeys: List of attribute names that are time-based
        attributes: Dictionary mapping attribute names to their specs
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "IrregularTimeSeries"
    length: int
    domain_start: float
    domain_end: float
    timekeys: List[str]
    attributes: Dict[str, AttributeSpec]


class RegularTimeSeriesSpec(BaseModel):
    """Specification for a RegularTimeSeries object.

    RegularTimeSeries represents uniformly sampled time series data.

    Attributes:
        type: Always "RegularTimeSeries"
        length: Number of samples
        sampling_rate: Sampling rate in Hz
        domain_start: Start time of the domain
        domain_end: End time of the domain
        attributes: Dictionary mapping attribute names to their specs
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "RegularTimeSeries"
    length: int
    sampling_rate: float
    domain_start: float
    domain_end: float
    attributes: Dict[str, AttributeSpec]


# Type alias for any spec that can be nested within a DataSpec
NestedSpec = Union[
    ArrayDictSpec,
    IntervalSpec,
    IrregularTimeSeriesSpec,
    RegularTimeSeriesSpec,
    AttributeSpec,
    ScalarSpec,
    "DataSpec",
]


class DataSpec(BaseModel):
    """Specification for a Data container object.

    Data is the top-level container that can hold various temporal data objects
    and metadata.

    Attributes:
        type: Always "Data"
        domain_start: Start time of the domain (None if no temporal data)
        domain_end: End time of the domain (None if no temporal data)
        attributes: Dictionary mapping attribute names to their specs
            (can contain nested DataSpec, ArrayDictSpec, etc.)
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "Data"
    domain_start: Optional[float] = None
    domain_end: Optional[float] = None
    attributes: Dict[str, Any]  # Can contain nested specs or scalar types


# Export all spec classes
__all__ = [
    "AttributeSpec",
    "ScalarSpec",
    "ArrayDictSpec",
    "IntervalSpec",
    "IrregularTimeSeriesSpec",
    "RegularTimeSeriesSpec",
    "DataSpec",
    "NestedSpec",
]
