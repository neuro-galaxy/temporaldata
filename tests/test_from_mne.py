import os
from pathlib import Path

import numpy as np
import pytest
import mne

from temporaldata import Data, RegularTimeSeries, IrregularTimeSeries, Interval

from temporaldata.io.from_mne import (
    raw_to_temporaldata,
    epochs_to_temporaldata,
    epochs_to_temporaldata_lazy,
)


def _make_synthetic_raw(with_annotations: bool = False) -> mne.io.Raw:
    """Create a small synthetic Raw object for testing."""
    sfreq = 100.0
    n_channels = 4
    n_times = 1000
    data = np.random.randn(n_channels, n_times)
    ch_names = [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    if with_annotations:
        onset = np.array([0.5, 2.0])
        duration = np.array([0.2, 0.3])
        description = np.array(["A", "B"], dtype=object)
        ann = mne.Annotations(onset=onset, duration=duration, description=description)
        raw.set_annotations(ann)

    return raw


def test_raw_to_temporaldata_eager_basic():
    raw = _make_synthetic_raw(with_annotations=False)

    td = raw_to_temporaldata(raw, data_key="eeg")
    sfreq = float(raw.info["sfreq"])

    # Container type
    assert isinstance(td, Data)

    # Continuous signal
    # Attribute-based access
    assert hasattr(td, "eeg")
    signal = td.eeg
    assert isinstance(signal, RegularTimeSeries)

    # Shape: (n_times, n_channels)
    n_channels = raw.info["nchan"]
    n_times = raw.n_times
    assert signal.raw.shape == (n_times, n_channels)

    # Recompute times from Raw
    _, times = raw.get_data(return_times=True)
    start = float(times[0])
    end = float(times[-1])

    # Sampling rate and domain
    # assert np.isclose(signal.sampling_rate, sfreq)
    assert pytest.approx(signal.sampling_rate) == sfreq
    assert pytest.approx(signal.domain.start) == start
    assert pytest.approx(signal.domain.end) == end

    # Metadata keys
    assert td.sfreq == pytest.approx(float(sfreq))
    assert td.n_channels == raw.info["nchan"]
    assert td.n_times == raw.n_times
    assert td.meas_date == raw.info["meas_date"]
    np.testing.assert_array_equal(
        td.channel_names, np.array(raw.ch_names, dtype=object)
    )


def test_raw_to_temporaldata_eager_with_events():
    raw = _make_synthetic_raw(with_annotations=True)
    sfreq = float(raw.info["sfreq"])

    # Ground truth events from MNE
    events_from_ann, event_id = mne.events_from_annotations(raw)
    assert events_from_ann.shape[0] > 0  # sanity

    td = raw_to_temporaldata(raw, data_key="eeg")

    # events and event_id should be present as attributes
    assert hasattr(td, "events")
    assert hasattr(td, "event_id")

    ev = td.events
    assert isinstance(ev, IrregularTimeSeries)

    # Timestamps in seconds
    expected_timestamps = events_from_ann[:, 0] / sfreq
    assert ev.timestamps.shape == expected_timestamps.shape
    assert np.allclose(ev.timestamps, expected_timestamps)

    # Event codes: last column
    expected_codes = events_from_ann[:, -1].astype("int64")
    assert ev.event_code.shape == expected_codes.shape
    assert np.array_equal(ev.event_code, expected_codes)

    # event_id dict should match
    assert td.event_id == event_id


def test_raw_to_temporaldata_eager_with_annotations():
    # Raw with two annotations A and B
    raw = _make_synthetic_raw(with_annotations=True)
    ann = raw.annotations
    assert len(ann) > 0  # sanity

    td = raw_to_temporaldata(raw, data_key="eeg")

    # Data container
    assert isinstance(td, Data)

    # annotations attribute should exist
    assert hasattr(td, "annotations")
    intervals = td.annotations
    assert isinstance(intervals, Interval)

    # Expected start/end/labels from the Raw annotations
    starts_expected = np.asarray(ann.onset, dtype=float)
    ends_expected = starts_expected + np.asarray(ann.duration, dtype=float)
    labels_expected = np.asarray(ann.description, dtype=object)

    # Compare Interval contents
    assert np.allclose(intervals.start, starts_expected)
    assert np.allclose(intervals.end, ends_expected)
    assert np.array_equal(intervals.label, labels_expected)


def _make_synthetic_epochs(
    n_epochs=5,
    n_channels=3,
    n_times=10,
    sfreq=100.0,
    tmin=-0.1,
    event_id=None,
):
    """Helper: build a tiny, fully-controlled MNE Epochs object."""
    rng = np.random.RandomState(0)
    data = rng.randn(n_epochs, n_channels, n_times).astype(np.float32)

    if event_id is None:
        event_id = {"stim/a": 1, "stim/b": 2}

    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )

    # Simple events: alternate between 1 and 2
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(n_epochs)
    events[:, 2] = np.tile([1, 2], n_epochs // 2 + 1)[:n_epochs]

    epochs = mne.EpochsArray(
        data,
        info=info,
        events=events,
        event_id=event_id,
        tmin=tmin,
        verbose=False,
    )
    return epochs, data, events, event_id


def test_epochs_to_temporaldata_eager_basic():
    epochs, data, events, event_id = _make_synthetic_epochs()

    td = epochs_to_temporaldata(epochs, data_key="eeg", trials_key="trials")

    # Basic structure
    assert isinstance(td, Data)
    assert hasattr(td, "eeg")
    assert hasattr(td, "trials")

    rts = td.eeg
    trials = td.trials

    assert isinstance(rts, RegularTimeSeries)
    assert isinstance(trials, Interval)

    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info["sfreq"]
    epoch_dur = n_times / sfreq

    # Shape of time series: (T, C) with T = n_epochs * n_times
    assert rts.raw.shape == (n_epochs * n_times, n_channels)

    # Check sampling rate and domain
    assert pytest.approx(rts.sampling_rate) == sfreq
    assert pytest.approx(rts.domain.start) == epochs.tmin
    assert pytest.approx(rts.domain.end) == epochs.tmin + n_epochs * epoch_dur

    # Check trial Interval: starts and ends are contiguous and correct
    assert trials.start.shape == (n_epochs,)
    assert trials.end.shape == (n_epochs,)
    assert pytest.approx(trials.start[0]) == epochs.tmin
    for i in range(n_epochs):
        assert pytest.approx(trials.end[i] - trials.start[i]) == epoch_dur
    # consecutive
    assert np.allclose(trials.start[1:], trials.end[:-1])

    # Event codes should match epochs.events[:, 2]
    assert np.array_equal(trials.event_code, events[:, 2])

    # Event names mapping: code -> label from event_id
    code_to_name = {v: k for k, v in event_id.items()}
    if hasattr(trials, "event_name"):
        expected_names = np.array([code_to_name[c] for c in events[:, 2]], dtype=object)
        assert np.array_equal(trials.event_name, expected_names)

    # Channel names
    assert np.array_equal(td.channel_names, np.array(epochs.ch_names, dtype=object))

    # Data values: check exact reshape correspondence
    # epochs: (E, C, T)  -> (E, T, C) -> (E*T, C)
    expected = np.swapaxes(data, 1, 2).reshape(-1, n_channels)
    np.testing.assert_allclose(rts.raw, expected, rtol=0, atol=1e-6)


def test_epochs_to_temporaldata_lazy_basic(tmp_path):
    epochs, data, events, event_id = _make_synthetic_epochs(
        n_epochs=7
    )  # non-multiple of batch

    mmap_path = tmp_path / "eeg.mm"
    td = epochs_to_temporaldata_lazy(
        epochs,
        mmap_path=mmap_path,
        data_key="eeg",
        trials_key="trials",
        batch_size=3,  # intentionally not dividing n_epochs
        overwrite=True,
    )

    # File exists on disk
    assert mmap_path.exists()

    rts = td.eeg
    trials = td.trials

    assert isinstance(rts, RegularTimeSeries)
    assert isinstance(trials, Interval)

    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info["sfreq"]
    epoch_dur = n_times / sfreq

    # Shape of memmapped time series
    assert rts.raw.shape == (n_epochs * n_times, n_channels)

    # dtype
    assert isinstance(rts.raw, np.memmap)
    assert rts.raw.dtype == np.float32

    # Sampling rate and domain
    assert pytest.approx(rts.sampling_rate) == sfreq
    assert pytest.approx(rts.domain.start) == epochs.tmin
    assert pytest.approx(rts.domain.end) == epochs.tmin + n_epochs * epoch_dur

    # Trials metadata
    assert trials.start.shape == (n_epochs,)
    assert trials.end.shape == (n_epochs,)
    assert np.array_equal(trials.event_code, events[:, 2])

    # Event names
    code_to_name = {v: k for k, v in event_id.items()}
    if hasattr(trials, "event_name"):
        expected_names = np.array([code_to_name[c] for c in events[:, 2]], dtype=object)
        assert np.array_equal(trials.event_name, expected_names)

    # Channel names
    assert np.array_equal(td.channel_names, np.array(epochs.ch_names, dtype=object))

    # Data values: memmap content should match eager reshape of original
    expected = np.swapaxes(data, 1, 2).reshape(-1, n_channels)
    np.testing.assert_allclose(np.asarray(rts.raw), expected, rtol=0, atol=1e-6)

    # Metadata path
    assert td.raw_mmap_path == str(mmap_path)


def test_epochs_to_temporaldata_lazy_overwrite_flag(tmp_path):
    epochs, data, events, event_id = _make_synthetic_epochs()

    mmap_path = tmp_path / "eeg.mm"
    mmap_path.write_bytes(b"dummy")  # create file

    # overwrite=False should raise
    with pytest.raises(FileExistsError):
        epochs_to_temporaldata_lazy(
            epochs,
            mmap_path=mmap_path,
            overwrite=False,
        )

    # overwrite=True should succeed
    td = epochs_to_temporaldata_lazy(
        epochs,
        mmap_path=mmap_path,
        overwrite=True,
    )
    assert mmap_path.exists()
    assert isinstance(td.eeg.raw, np.memmap)
