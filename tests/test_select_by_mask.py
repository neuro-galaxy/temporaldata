"""Parametrized tests for the input-validation failure modes of
``select_by_mask`` across all four data classes and their Lazy variants.

Each ``select_by_mask`` implementation re-asserts the same three invariants:
mask must be 1D, mask must be boolean, mask length must match the first
dimension. These tests verify the exception is raised on every class.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from temporaldata import (
    ArrayDict,
    Interval,
    IrregularTimeSeries,
    LazyArrayDict,
    LazyInterval,
    LazyIrregularTimeSeries,
)


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name
    tmpfile.close()

    def finalizer():
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


def _make_array_dict():
    return ArrayDict(x=np.array([1, 2, 3]))


def _make_interval():
    return Interval(
        start=np.array([0.0, 1.0, 2.0]),
        end=np.array([1.0, 2.0, 3.0]),
    )


def _make_irregular():
    return IrregularTimeSeries(
        timestamps=np.array([0.1, 0.2, 0.3]),
        domain="auto",
    )


def _make_lazy(non_lazy, lazy_cls, test_filepath):
    with h5py.File(test_filepath, "w") as f:
        non_lazy.to_hdf5(f)
    f = h5py.File(test_filepath, "r")
    return lazy_cls.from_hdf5(f), f


@pytest.fixture(
    params=[
        "ArrayDict",
        "Interval",
        "IrregularTimeSeries",
        "LazyArrayDict",
        "LazyInterval",
        "LazyIrregularTimeSeries",
    ]
)
def obj(request, test_filepath):
    name = request.param
    if name == "ArrayDict":
        yield _make_array_dict()
    elif name == "Interval":
        yield _make_interval()
    elif name == "IrregularTimeSeries":
        yield _make_irregular()
    elif name == "LazyArrayDict":
        instance, f = _make_lazy(_make_array_dict(), LazyArrayDict, test_filepath)
        yield instance
        f.close()
    elif name == "LazyInterval":
        instance, f = _make_lazy(_make_interval(), LazyInterval, test_filepath)
        yield instance
        f.close()
    elif name == "LazyIrregularTimeSeries":
        instance, f = _make_lazy(
            _make_irregular(), LazyIrregularTimeSeries, test_filepath
        )
        yield instance
        f.close()


def test_select_by_mask_rejects_2d_mask(obj):
    with pytest.raises(ValueError, match="mask must be 1D"):
        obj.select_by_mask(np.array([[True, False, True]]))


def test_select_by_mask_rejects_non_bool_mask(obj):
    with pytest.raises(ValueError, match="mask must be boolean"):
        obj.select_by_mask(np.array([0, 1, 1]))


def test_select_by_mask_rejects_length_mismatch(obj):
    with pytest.raises(ValueError, match="does not match first dimension"):
        obj.select_by_mask(np.array([True, False]))


class TestLazyMaskIsCopied:

    def test_lazy_arraydict(self, test_filepath):
        data, f = _make_lazy(_make_array_dict(), LazyArrayDict, test_filepath)
        mask = np.array([True, False, True])
        masked = data.select_by_mask(mask)
        # modify mask. `masked` should NOT care about this
        mask[0] = False
        assert len(masked.x) == 2

    def test_lazy_arraydict_doublemask(self, test_filepath):
        data, f = _make_lazy(_make_array_dict(), LazyArrayDict, test_filepath)
        mask1 = np.array([True, False, True])
        masked = data.select_by_mask(mask1)
        mask2 = np.array([True, False])
        masked2 = masked.select_by_mask(mask2)
        mask1[0] = False
        assert len(masked2.x) == 1

    def test_lazy_irregular_ts(self, test_filepath):
        data, f = _make_lazy(_make_irregular(), LazyIrregularTimeSeries, test_filepath)
        mask = np.array([True, False, True])
        masked = data.select_by_mask(mask)
        # modify mask. `masked` should NOT care about this
        mask[0] = False
        assert len(masked.timestamps) == 2

    def test_lazy_irregular_ts_doublemask(self, test_filepath):
        data, f = _make_lazy(_make_irregular(), LazyIrregularTimeSeries, test_filepath)
        mask1 = np.array([True, False, True])
        masked = data.select_by_mask(mask1)
        mask2 = np.array([True, False])
        masked2 = masked.select_by_mask(mask2)
        mask1[0] = False
        assert len(masked2.timestamps) == 1

    def test_lazy_interval(self, test_filepath):
        data, f = _make_lazy(_make_interval(), LazyInterval, test_filepath)
        mask = np.array([True, False, True])
        masked = data.select_by_mask(mask)
        # modify mask. `masked` should NOT care about this
        mask[0] = False
        assert len(masked.start) == 2

    def test_lazy_interval_doublemask(self, test_filepath):
        data, f = _make_lazy(_make_interval(), LazyInterval, test_filepath)
        mask1 = np.array([True, False, True])
        masked = data.select_by_mask(mask1)
        mask2 = np.array([True, False])
        masked2 = masked.select_by_mask(mask2)
        mask1[0] = False
        assert len(masked2.start) == 1
