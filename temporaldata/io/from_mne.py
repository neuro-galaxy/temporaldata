from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import mne
from ..temporaldata import Data, IrregularTimeSeries, RegularTimeSeries, Interval

import os
from uuid import uuid4


def raw_to_temporaldata(
    raw: mne.io.Raw,
    data_key: str = "eeg",
) -> Data:
    """
    Convert an MNE-Python raw object to a temporaldata.Data container (eager).

    This version calls `raw.get_data()` and materializes the full
    array in memory. It is simple and fast for small/mid-sized datasets, but
    can be memory-hungry for large recordings.

    Parameters
    ----------
    raw : mne.io.Raw
        The continuous MNE-Python Raw object to convert. It must contain
        the data and metadata in `raw.get_data()`, `raw.info`, and
        `raw.annotations`.
    data_key : str, optional
        Key under which the continuous RegularTimeSeries will be stored
        in the returned Data object (default is "eeg").

    Returns
    -------
    temporal_data : temporaldata.Data
        A Data object containing:
          - `data_key` : RegularTimeSeries of continuous data.
          - `meas_date` : Measurement start datetime from `raw.info["meas_date"]`.
          - `sfreq` : Sampling frequency in Hz.
          - `channel_names` : Array of channel name strings.
          - `n_channels` : Number of channels.
          - `n_times`: Number of time points.
          - `domain` : Domain specifier (typically "auto").
          - `events` (optional): IrregularTimeSeries of events (if events could be parsed from annotations).
          - `event_id` (optional): dict mapping event labels to integer codes (if events are present).
          - `annotations` (optional): Interval of annotation segments (if annotations are present).
    """

    # Continuous data -> RegularTimeSeries
    raw_data, times = raw.get_data(return_times=True)  # (n_channels, n_times)
    raw_data = raw_data.T  # (n_times, n_channels)

    sfreq = float(raw.info["sfreq"])

    signal = RegularTimeSeries(
        raw=raw_data,
        sampling_rate=sfreq,
        domain_start=times[0],
        domain="auto",
    )

    # Events -> IrregularTimeSeries
    td_events = None
    event_id = None

    events, event_id = mne.events_from_annotations(raw)

    if events is not None and len(events) > 0:
        td_events = IrregularTimeSeries(
            timestamps=events[:, 0] / sfreq,
            event_code=events[:, -1].astype("int64"),
            domain="auto",
        )

    # Annotations -> Interval
    td_ann = None
    ann = raw.annotations
    if ann is not None and len(ann) > 0:
        starts = np.asarray(ann.onset, dtype=float)  # shape: (n_ann,)
        ends = starts + np.asarray(ann.duration, dtype=float)  # shape: (n_ann,)
        labels = np.asarray(ann.description, dtype=object)  # shape: (n_ann,)

        interval_kwargs = {
            "start": starts,
            "end": ends,
            "label": labels,
            "timekeys": ["start", "end"],
        }

        td_ann = Interval(**interval_kwargs)

    data_kwargs = {
        data_key: signal,
        "meas_date": raw.info["meas_date"],
        "sfreq": sfreq,
        "channel_names": np.array(raw.ch_names, dtype=object),
        "n_channels": raw.info["nchan"],
        "n_times": raw.n_times,
        "domain": "auto",
    }

    if td_events is not None:
        data_kwargs["events"] = td_events
    if event_id is not None:
        data_kwargs["event_id"] = event_id
    if td_ann is not None:
        data_kwargs["annotations"] = td_ann

    temporal_data = Data(**data_kwargs)

    return temporal_data


def raw_to_temporaldata_lazy(
    raw: mne.io.Raw,
    *,
    mmap_path: Union[str, Path] = None,
    data_key: str = "eeg",
    batch_size: int = 8,
    dtype: Union[str, np.dtype] = np.float32,
    overwrite: bool = False,
) -> Data:
    """
    Convert an MNE-Python raw object to a temporaldata.Data container (lazy).
    """

    if mmap_path is None:
        base = Path.home() / ".temporaldata" / "memmap"
        base.mkdir(parents=True, exist_ok=True)
        mmap_path = base / f"mne_raw_{uuid4().hex}.mm"

    mmap_path = Path(mmap_path)

    if mmap_path.exists() and not overwrite:
        raise FileExistsError(
            f"{mmap_path} already exists. Set overwrite=True to overwrite."
        )

    sfreq = float(raw.info["sfreq"])
    n_channels = raw.info["nchan"]
    n_times = raw.n_times

    t_start = float(raw.first_samp) / sfreq
    t_end = t_start + n_times / sfreq

    channel_names = np.array(raw.ch_names, dtype=object)
    meas_date = raw.info["meas_date"]

    # ------------------------------------------------------------------
    # Create memmap on disk and stream epochs into it in small batches
    # ------------------------------------------------------------------
    mmap = np.memmap(
        mmap_path,
        mode="w+",
        dtype=dtype,
        shape=(n_times, n_channels),
    )

    write_offset = 0
    batch_size = max(1, int(batch_size))  # safety

    for start in range(0, n_times, batch_size):
        stop = min(start + batch_size, n_times)

        # raw.get_data returns (n_channels, n_times_chunk)
        chunk = raw.get_data(start=start, stop=stop)

        # -> (chunk_len, n_channels)
        chunk_t = chunk.T
        if chunk_t.dtype != dtype:
            chunk_t = chunk_t.astype(dtype, copy=False)

        chunk_len = chunk_t.shape[0]
        mmap[write_offset : write_offset + chunk_len, :] = chunk_t
        write_offset += chunk_len

    mmap.flush()

    # ------------------------------------------------------------------
    # Build RegularTimeSeries over continuous time axis
    # ------------------------------------------------------------------
    domain = Interval(start=t_start, end=t_end)

    signal = RegularTimeSeries(
        raw=mmap,
        sampling_rate=sfreq,
        domain=domain,
    )

    # ------------------------------------------------------------------
    # Events from annotations (optional, small → eager)
    # ------------------------------------------------------------------
    td_events = None
    event_id = None

    events, event_id = mne.events_from_annotations(raw)

    if events is not None and len(events) > 0:
        td_events = IrregularTimeSeries(
            timestamps=events[:, 0] / sfreq,
            event_code=events[:, -1].astype("int64"),
            domain="auto",
        )

    # ------------------------------------------------------------------
    # Annotations → Interval (optional, small)
    # ------------------------------------------------------------------
    td_ann = None
    ann = raw.annotations
    if ann is not None and len(ann) > 0:
        starts = np.asarray(ann.onset, dtype=float)
        ends = starts + np.asarray(ann.duration, dtype=float)
        labels = np.asarray(ann.description, dtype=object)
        td_ann = Interval(
            start=starts,
            end=ends,
            label=labels,
            timekeys=["start", "end"],
        )

    # ------------------------------------------------------------------
    # Assemble Data container
    # ------------------------------------------------------------------
    data_kwargs = {
        data_key: signal,
        "meas_date": meas_date,
        "sfreq": sfreq,
        "channel_names": channel_names,
        "n_channels": n_channels,
        "n_times": n_times,
        "domain": domain,
        "raw_mmap_path": str(mmap_path),
    }

    if td_events is not None:
        data_kwargs["events"] = td_events
    if event_id:
        data_kwargs["event_id"] = event_id
    if td_ann is not None:
        data_kwargs["annotations"] = td_ann

    temporal_data = Data(**data_kwargs)

    return temporal_data


def epochs_to_temporaldata(
    epochs: mne.Epochs,
    *,
    data_key: str = "eeg",
    trials_key: str = "trials",
    include_event_names: bool = True,
    copy: bool = True,
    dtype: Union[str, np.dtype] = None,
) -> Data:
    """
    Convert an MNE Epochs object to a temporaldata.Data container (eager).

    This version calls `epochs.get_data(copy=...)` and materializes the full
    array in memory. It is simple and fast for small/mid-sized datasets, but
    can be memory-hungry for large recordings.

    The returned Data object has:
      - A RegularTimeSeries at `data_key` containing the continuous
        concatenation of all epochs.
      - An Interval at `trials_key` describing the start / end of each epoch
        in that concatenated time axis, plus event metadata.

    Parameters
    ----------
    epochs : mne.Epochs
        The MNE epochs object to convert.
    data_key : str, optional
        Attribute name under which the RegularTimeSeries will be stored
        inside the returned Data object. Default is "eeg".
    trials_key : str, optional
        Attribute name under which the Interval (trial boundaries +
        metadata) will be stored. Default is "trials".
    include_event_names : bool, optional
        If True, an `event_name` attribute is added to the Interval,
        derived from `epochs.event_id`. Default is True.
    copy : bool, optional
        Passed to `epochs.get_data(copy=...)`. Default is True.
    dtype : str or np.dtype, optional
        If provided, the time series data will be cast to this dtype before
        creating the RegularTimeSeries. Useful to downcast to float32.

    Returns
    -------
    temporal_data : temporaldata.Data
        A Data object containing:
          - `data_key` : RegularTimeSeries (shape (T, C))
          - `trials_key` : Interval with trial metadata
          - `channel_names`, `sfreq`, `tmin`, `tmax`, shape metadata
          - `domain` : Interval for the concatenated time axis
    """
    sfreq = float(epochs.info["sfreq"])
    tmin = float(epochs.tmin)
    tmax = float(epochs.tmax)

    # data: (n_epochs, n_channels, n_times)
    X = epochs.get_data(copy=copy)
    n_epochs, n_channels, n_times = X.shape
    epoch_duration = n_times / sfreq

    if dtype is not None and X.dtype != dtype:
        X = X.astype(dtype, copy=False)

    # MNE: (n_epochs, n_channels, n_times)
    # We want: time-major (T, C) where T = n_epochs * n_times
    X_t = np.swapaxes(X, 1, 2).reshape(n_epochs * n_times, n_channels)

    # Time domain: epochs concatenated back-to-back starting at tmin
    domain = Interval(
        start=tmin,
        end=tmin + n_epochs * epoch_duration,
        timekeys=["start", "end"],
    )

    signal = RegularTimeSeries(
        raw=X_t,
        sampling_rate=sfreq,
        domain=domain,
    )

    # Trial / epoch metadata as Interval
    starts = tmin + np.arange(n_epochs) * epoch_duration
    ends = starts + epoch_duration

    event_codes = epochs.events[:, 2].astype("int64")

    interval_kwargs = {
        "start": starts,
        "end": ends,
        "event_code": event_codes,
        "timekeys": ["start", "end"],
    }

    if include_event_names:
        code_to_name = {v: k for k, v in epochs.event_id.items()}
        event_names = np.array(
            [code_to_name.get(code, "") for code in event_codes],
            dtype=object,
        )
        interval_kwargs["event_name"] = event_names

    trials = Interval(**interval_kwargs)

    temporal_data = Data(
        **{
            data_key: signal,
            trials_key: trials,
            "channel_names": np.array(epochs.ch_names, dtype=object),
            "sfreq": sfreq,
            "tmin": tmin,
            "tmax": tmax,
            "n_epochs": n_epochs,
            "n_channels": n_channels,
            "n_times": n_times,
            "domain": domain,
        }
    )

    return temporal_data


def epochs_to_temporaldata_lazy(
    epochs: mne.Epochs,
    *,
    mmap_path: Union[str, Path] = None,
    data_key: str = "eeg",
    trials_key: str = "trials",
    batch_size: int = 8,
    dtype: Union[str, np.dtype] = np.float32,
    include_event_names: bool = True,
    overwrite: bool = False,
) -> Data:
    """
    Convert an MNE Epochs object into a temporaldata.Data container **without**
    ever materializing all epochs in memory at once.

    Data are streamed in small batches into a NumPy memmap on disk, which is
    then wrapped by a RegularTimeSeries inside a Data object.

    Parameters
    ----------
    epochs
        MNE Epochs object to convert. Assumed to have fixed-length epochs.
    mmap_path
        Path to the memmapped file that will store the concatenated
        (T, C) time series. Will be created (or overwritten if
        ``overwrite=True``).
    data_key
        Attribute name for the RegularTimeSeries inside the returned
        Data object. Default: "eeg".
    trials_key
        Attribute name for the Interval describing trial boundaries and
        event metadata. Default: "trials".
    batch_size
        Number of epochs to load from MNE at once. Controls peak RAM
        usage during conversion. Default: 8.
    dtype
        Data type for the memmap and resulting time series. Default:
        ``np.float32``.
    include_event_names
        If True, add an ``event_name`` field to the Interval, mapping
        integer event codes to their names using ``epochs.event_id``.
        Default: True.
    overwrite
        If False and ``mmap_path`` already exists, raise a
        FileExistsError. If True, overwrite any existing file.
        Default: False.

    Returns
    -------
    data
        temporaldata.Data object with:
          - ``{data_key}``: RegularTimeSeries backed by a memmap of
            shape (T, C), where T = n_epochs * n_times.
          - ``{trials_key}``: Interval with trial ``start`` / ``end``
            times and event metadata.
          - basic metadata fields (channel_names, sfreq, tmin, tmax,
            n_epochs, n_channels, n_times, domain, raw_mmap_path).

    Notes
    -----
    The main purpose of this converter is to remove the memory spike
    associated with ``epochs.get_data()`` on large datasets. After
    running this function and constructing the Data object, you can
    safely delete the original ``epochs`` object to reclaim memory:

    .. code-block:: python

        td = epochs_to_temporaldata_lazy(epochs, mmap_path="eeg.mm")
        del epochs
        import gc; gc.collect()

    Downstream pipelines can then operate purely on temporaldata.
    """
    if mmap_path is None:
        base = Path.home() / ".temporaldata" / "memmap"
        base.mkdir(parents=True, exist_ok=True)
        mmap_path = base / f"mne_{uuid4().hex}.mm"

    mmap_path = Path(mmap_path)

    if mmap_path.exists() and not overwrite:
        raise FileExistsError(
            f"{mmap_path} already exists. Set overwrite=True to overwrite."
        )

    sfreq = float(epochs.info["sfreq"])
    tmin = float(epochs.tmin)
    tmax = float(epochs.tmax)

    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    n_times = len(epochs.times)  # samples per epoch

    epoch_duration = n_times / sfreq  # seconds per epoch
    total_T = n_epochs * n_times  # total time samples after concat

    # ------------------------------------------------------------------
    # Create memmap on disk and stream epochs into it in small batches
    # ------------------------------------------------------------------
    mmap = np.memmap(
        mmap_path,
        mode="w+",
        dtype=dtype,
        shape=(total_T, n_channels),
    )

    write_offset = 0
    for start in range(0, n_epochs, batch_size):
        stop = min(start + batch_size, n_epochs)

        # Load only this batch of epochs from MNE
        # shape: (batch, n_channels, n_times)
        batch = epochs.get_data(item=slice(start, stop))

        # (batch, n_channels, n_times) -> (batch, n_times, n_channels)
        #                               -> (batch * n_times, n_channels)
        batch_t = np.swapaxes(batch, 1, 2).reshape(-1, n_channels)
        if batch_t.dtype != dtype:
            batch_t = batch_t.astype(dtype, copy=False)

        batch_T = batch_t.shape[0]
        mmap[write_offset : write_offset + batch_T, :] = batch_t
        write_offset += batch_T

    mmap.flush()

    # ------------------------------------------------------------------
    # Build RegularTimeSeries over concatenated time axis
    # ------------------------------------------------------------------
    domain = Interval(start=tmin, end=tmin + n_epochs * epoch_duration)

    signal = RegularTimeSeries(
        raw=mmap,
        sampling_rate=sfreq,
        domain=domain,
    )

    # ------------------------------------------------------------------
    # Trial / epoch metadata as Interval
    # ------------------------------------------------------------------
    starts = tmin + np.arange(n_epochs) * epoch_duration
    ends = starts + epoch_duration

    event_codes = epochs.events[:, 2].astype("int64")

    interval_kwargs = {
        "start": starts,
        "end": ends,
        "event_code": event_codes,
        "timekeys": ["start", "end"],
    }

    if include_event_names:
        code_to_name = {v: k for k, v in epochs.event_id.items()}
        event_names = np.array(
            [code_to_name.get(code, "") for code in event_codes],
            dtype=object,
        )
        interval_kwargs["event_name"] = event_names

    trials = Interval(**interval_kwargs)

    # ------------------------------------------------------------------
    # Assemble Data container
    # ------------------------------------------------------------------
    data = Data(
        **{
            data_key: signal,
            trials_key: trials,
            "channel_names": np.array(epochs.ch_names, dtype=object),
            "sfreq": sfreq,
            "tmin": tmin,
            "tmax": tmax,
            "n_epochs": n_epochs,
            "n_channels": n_channels,
            "n_times": n_times,
            "domain": domain,
            "raw_mmap_path": str(mmap_path),
        }
    )

    return data
