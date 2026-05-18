import logging
import os
import tempfile

import h5py
import numpy as np
import pytest

from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    LazyArrayDict,
    LazyInterval,
    LazyIrregularTimeSeries,
    LazyRegularTimeSeries,
    RegularTimeSeries,
    resolve_on_access,
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


@pytest.fixture
def saved_data(test_filepath):
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.0, 1.0, 2.0]),
            values=np.array([10.0, 20.0, 30.0]),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.random.random((100, 4)),
            sampling_rate=10,
            domain=Interval(0.0, 10.0),
        ),
        trials=Interval(
            start=np.array([0.0, 1.0, 2.0]),
            end=np.array([1.0, 2.0, 3.0]),
        ),
        extras=ArrayDict(x=np.array([1, 2, 3])),
        domain=Interval(0.0, 3.0),
    )
    data.save(test_filepath)
    return test_filepath


class TestAutoresolveOff:
    """Accessing lazy attributes with resolve_on_access(False) returns h5py.Dataset."""

    def test_irregular_timeseries(self, saved_data):
        with Data.load(saved_data) as data:
            assert isinstance(data.spikes, LazyIrregularTimeSeries)
            with resolve_on_access(False):
                assert isinstance(data.spikes.values, h5py.Dataset)
                assert isinstance(data.spikes.timestamps, h5py.Dataset)
            # object should still be lazy — nothing was cached
            assert isinstance(data.spikes.__dict__["values"], h5py.Dataset)

    def test_regular_timeseries(self, saved_data):
        with Data.load(saved_data) as data:
            assert isinstance(data.lfp, LazyRegularTimeSeries)
            with resolve_on_access(False):
                assert isinstance(data.lfp.raw, h5py.Dataset)
            assert isinstance(data.lfp.__dict__["raw"], h5py.Dataset)

    def test_interval(self, saved_data):
        with Data.load(saved_data) as data:
            assert isinstance(data.trials, LazyInterval)
            with resolve_on_access(False):
                assert isinstance(data.trials.start, h5py.Dataset)
                assert isinstance(data.trials.end, h5py.Dataset)
            assert isinstance(data.trials.__dict__["start"], h5py.Dataset)

    def test_arraydict(self, saved_data):
        with Data.load(saved_data) as data:
            assert isinstance(data.extras, LazyArrayDict)
            with resolve_on_access(False):
                assert isinstance(data.extras.x, h5py.Dataset)
            assert isinstance(data.extras.__dict__["x"], h5py.Dataset)


class TestAutoresolveOn:
    """With resolve_on_access(True) (or default), normal resolution happens."""

    def test_default_resolves(self, saved_data):
        with Data.load(saved_data) as data:
            vals = data.spikes.values
            assert isinstance(vals, np.ndarray)
            np.testing.assert_array_equal(vals, [10.0, 20.0, 30.0])

    def test_explicit_true_resolves(self, saved_data):
        with Data.load(saved_data) as data:
            with resolve_on_access(True):
                vals = data.spikes.values
                assert isinstance(vals, np.ndarray)


class TestAutoresolveNesting:
    """Nested context managers restore correctly."""

    def test_nested_off_on(self, saved_data):
        with Data.load(saved_data) as data:
            with resolve_on_access(False):
                assert isinstance(data.spikes.values, h5py.Dataset)
                with resolve_on_access(True):
                    # inner context enables resolution
                    vals = data.spikes.values
                    assert isinstance(vals, np.ndarray)

    def test_nested_restore(self, saved_data):
        with Data.load(saved_data) as data:
            with resolve_on_access(False):
                pass
            # after exiting, default (True) is restored
            vals = data.spikes.values
            assert isinstance(vals, np.ndarray)


class TestAutoresolveNoClassUpgrade:
    """With resolve_on_access(False), lazy class is never swapped to non-lazy."""

    def test_no_upgrade_after_accessing_all_keys(self, saved_data):
        with Data.load(saved_data) as data:
            with resolve_on_access(False):
                for key in data.spikes.keys():
                    getattr(data.spikes, key)
            # class should still be lazy
            assert isinstance(data.spikes, LazyIrregularTimeSeries)

    def test_upgrade_happens_normally(self, saved_data):
        with Data.load(saved_data) as data:
            for key in data.spikes.keys():
                getattr(data.spikes, key)
            assert isinstance(data.spikes, IrregularTimeSeries)


class TestAutoresolveWarnings:
    """resolve_on_access(False) warns when there are pending lazy ops."""

    def test_warns_on_pending_ops(self, saved_data, caplog):
        with Data.load(saved_data) as data:
            sliced = data.slice(0.5, 2.5)
            with resolve_on_access(False):
                with caplog.at_level(logging.WARNING):
                    sliced.spikes.values
                assert "pending lazy operations" in caplog.text

    def test_no_warning_without_ops(self, saved_data, caplog):
        with Data.load(saved_data) as data:
            with resolve_on_access(False):
                with caplog.at_level(logging.WARNING):
                    data.spikes.values
                assert "pending lazy operations" not in caplog.text
