"""Tests for the picklability of Lazy* types.

Background: ``h5py.Dataset`` / ``Group`` / ``File`` instances are explicitly
non-picklable -- they wrap a libhdf5 handle bound to the current process.
The Lazy* types in temporaldata defer h5py reads via ``__getattribute__``
interception, which means a sliced lazy object can still be carrying live
h5py refs in ``__dict__`` until somebody touches its attributes.

This is fine in single-process code, but ``torch.utils.data.DataLoader``
with ``num_workers > 0`` pickles every sample to ship it from the worker
process to the main process, and the lazy objects blow up there with
``TypeError: h5py objects cannot be pickled``.

Each Lazy* class therefore overrides ``__reduce__`` to call ``load()``
(materializing all h5py refs to numpy) before pickle captures state.
These tests verify that round-trip works and that multiprocessing-shipped
samples are still correct.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import tempfile

import h5py
import numpy as np
import pytest

from temporaldata import (
    Data,
    Interval,
    IrregularTimeSeries,
    LazyArrayDict,
    LazyInterval,
    LazyIrregularTimeSeries,
    LazyRegularTimeSeries,
    RegularTimeSeries,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lazy_data_file(tmp_path):
    """Create an HDF5 file containing a Data object with a member of every
    Lazy* class. Returns (path, expected_arrays)."""
    its = IrregularTimeSeries(
        timestamps=np.linspace(0.0, 9.0, 10).astype(np.float64),
        values=np.arange(10, dtype=np.float32) * 1.5,
        labels=np.array([f"l{i}" for i in range(10)]),
        domain="auto",
    )
    rts = RegularTimeSeries(
        signal=np.arange(100 * 4, dtype=np.float32).reshape(100, 4),
        sampling_rate=10.0,
        domain=Interval(0.0, 9.9),
    )
    iv = Interval(
        start=np.array([0.0, 2.5, 7.0]),
        end=np.array([1.0, 5.0, 9.0]),
        label=np.array(["a", "b", "c"]),
    )
    ad_data = Data(
        irregular=its,
        regular=rts,
        interval=iv,
        domain=Interval(0.0, 10.0),
    )

    path = tmp_path / "lazy.h5"
    with h5py.File(path, "w") as f:
        ad_data.to_hdf5(f)
    return path


# ---------------------------------------------------------------------------
# direct picklability of each Lazy* type
# ---------------------------------------------------------------------------


def test_lazy_irregular_ts_pickles(lazy_data_file):
    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)
        assert isinstance(loaded.irregular, LazyIrregularTimeSeries)

        roundtrip = pickle.loads(pickle.dumps(loaded.irregular))

    np.testing.assert_array_equal(
        roundtrip.timestamps, np.linspace(0.0, 9.0, 10).astype(np.float64)
    )
    np.testing.assert_array_equal(
        roundtrip.values, np.arange(10, dtype=np.float32) * 1.5
    )


def test_lazy_regular_ts_pickles(lazy_data_file):
    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)
        assert isinstance(loaded.regular, LazyRegularTimeSeries)

        roundtrip = pickle.loads(pickle.dumps(loaded.regular))

    np.testing.assert_array_equal(
        roundtrip.signal, np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    )


def test_lazy_interval_pickles(lazy_data_file):
    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)
        assert isinstance(loaded.interval, LazyInterval)

        roundtrip = pickle.loads(pickle.dumps(loaded.interval))

    np.testing.assert_array_equal(roundtrip.start, np.array([0.0, 2.5, 7.0]))
    np.testing.assert_array_equal(roundtrip.end, np.array([1.0, 5.0, 9.0]))


def test_lazy_arraydict_pickles(tmp_path):
    """LazyArrayDict round-trip via direct to/from_hdf5 (not wrapped in Data)."""
    from temporaldata import ArrayDict

    ad = ArrayDict(x=np.arange(7, dtype=np.float32), y=np.arange(7, dtype=np.int64))
    path = tmp_path / "ad.h5"
    with h5py.File(path, "w") as f:
        ad.to_hdf5(f)

    with h5py.File(path, "r") as f:
        loaded = LazyArrayDict.from_hdf5(f)
        roundtrip = pickle.loads(pickle.dumps(loaded))

    np.testing.assert_array_equal(roundtrip.x, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(roundtrip.y, np.arange(7, dtype=np.int64))


# ---------------------------------------------------------------------------
# the actual bug: deferred slice + pickle (worker scenario)
# ---------------------------------------------------------------------------


def test_lazy_irregular_ts_pickles_after_deferred_slice(lazy_data_file):
    """Slice a lazy object (which queues lazy ops without reading h5py),
    then pickle. This is the scenario that crashes a multi-worker DataLoader.
    """
    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)
        sliced = loaded.irregular.slice(2.0, 6.0, reset_origin=False)
        # Sanity: slice didn't materialize -- the values key should still be
        # an h5py.Dataset reference at this point.
        assert isinstance(sliced.__dict__.get("values"), h5py.Dataset), (
            "Test premise broken: slice unexpectedly materialized eagerly. "
            "If the slice impl changed to be eager, this test no longer "
            "exercises the deferred-h5py-ref bug."
        )
        roundtrip = pickle.loads(pickle.dumps(sliced))

    expected_ts = np.linspace(0.0, 9.0, 10)
    mask = (expected_ts >= 2.0) & (expected_ts < 6.0)
    np.testing.assert_array_equal(roundtrip.timestamps, expected_ts[mask])
    np.testing.assert_array_equal(
        roundtrip.values, (np.arange(10, dtype=np.float32) * 1.5)[mask]
    )


def test_lazy_data_full_tree_pickles(lazy_data_file):
    """Pickle the entire Data object containing every Lazy* variant."""
    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)
        roundtrip = pickle.loads(pickle.dumps(loaded))

    np.testing.assert_array_equal(
        roundtrip.irregular.timestamps,
        np.linspace(0.0, 9.0, 10).astype(np.float64),
    )
    np.testing.assert_array_equal(
        roundtrip.regular.signal,
        np.arange(100 * 4, dtype=np.float32).reshape(100, 4),
    )
    np.testing.assert_array_equal(
        roundtrip.interval.start, np.array([0.0, 2.5, 7.0])
    )


# ---------------------------------------------------------------------------
# multiprocessing: prove samples actually survive a process boundary
# ---------------------------------------------------------------------------


def _mp_worker_irregular(obj):
    """Worker function for multiprocessing pickling test. Returns a small
    summary so the parent can verify nothing was lost in transit."""
    return {
        "type": type(obj).__name__,
        "timestamps_sum": float(np.sum(obj.timestamps)),
        "values_sum": float(np.sum(obj.values)),
    }


def _mp_worker_data_tree(obj):
    return {
        "irregular_ts_sum": float(np.sum(obj.irregular.timestamps)),
        "regular_signal_sum": float(np.sum(obj.regular.signal)),
        "interval_start_sum": float(np.sum(obj.interval.start)),
    }


@pytest.mark.parametrize("start_method", ["spawn", "fork"])
def test_lazy_irregular_ts_through_multiprocessing(lazy_data_file, start_method):
    if start_method not in mp.get_all_start_methods():
        pytest.skip(f"start_method={start_method!r} not supported on this platform")

    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)
        sliced = loaded.irregular.slice(2.0, 6.0, reset_origin=False)

        ctx = mp.get_context(start_method)
        with ctx.Pool(1) as pool:
            (result,) = pool.map(_mp_worker_irregular, [sliced])

    expected_ts = np.linspace(0.0, 9.0, 10)
    mask = (expected_ts >= 2.0) & (expected_ts < 6.0)
    assert result["type"] in ("LazyIrregularTimeSeries", "IrregularTimeSeries")
    np.testing.assert_allclose(result["timestamps_sum"], expected_ts[mask].sum())
    np.testing.assert_allclose(
        result["values_sum"], (np.arange(10, dtype=np.float32) * 1.5)[mask].sum()
    )


@pytest.mark.parametrize("start_method", ["spawn", "fork"])
def test_lazy_data_tree_through_multiprocessing(lazy_data_file, start_method):
    if start_method not in mp.get_all_start_methods():
        pytest.skip(f"start_method={start_method!r} not supported on this platform")

    with h5py.File(lazy_data_file, "r") as f:
        loaded = Data.from_hdf5(f, lazy=True)

        ctx = mp.get_context(start_method)
        with ctx.Pool(1) as pool:
            (result,) = pool.map(_mp_worker_data_tree, [loaded])

    np.testing.assert_allclose(
        result["irregular_ts_sum"], np.linspace(0.0, 9.0, 10).sum()
    )
    np.testing.assert_allclose(
        result["regular_signal_sum"],
        np.arange(100 * 4, dtype=np.float32).reshape(100, 4).sum(),
    )
    np.testing.assert_allclose(
        result["interval_start_sum"], np.array([0.0, 2.5, 7.0]).sum()
    )


# ---------------------------------------------------------------------------
# DataLoader-style scenario (the original failure mode), if torch installed
# ---------------------------------------------------------------------------


def test_lazy_data_through_torch_dataloader_workers(lazy_data_file):
    """Mirrors the exact failure mode the user hit: DataLoader with
    num_workers > 0 pickling a sample containing Lazy* children.
    Skipped if torch isn't available."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, path):
            self.path = path

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            with h5py.File(self.path, "r") as f:
                d = Data.from_hdf5(f, lazy=True)
                # Slice: defers h5py reads. Pre-fix this would crash on
                # pickling at the worker -> main queue handoff.
                return d.irregular.slice(float(idx), float(idx) + 2.0,
                                         reset_origin=False)

    loader = DataLoader(
        _DS(str(lazy_data_file)),
        batch_size=1,
        num_workers=2,
        collate_fn=lambda batch: batch,  # no batching of custom types
    )

    samples = []
    for batch in loader:
        samples.extend(batch)

    assert len(samples) == 4
    for s in samples:
        # Each sample must be readable (no surviving h5py refs).
        assert s.timestamps.shape[0] >= 0
        assert s.values.shape[0] == s.timestamps.shape[0]
