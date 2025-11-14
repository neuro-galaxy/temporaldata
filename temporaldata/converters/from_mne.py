import mne

from temporaldata import IrregularTimeSeries, RegularTimeSeries, Data


def raw_to_temporaldata(raw: mne.io.Raw):
    """
    Convert an MNE-Python raw object to a temporaldata object
    """

    # Continuous data -> RegularTimeSeries
    raw_data, times = raw.get_data(return_times=True)
    raw_data = raw_data.T
    td_raw = RegularTimeSeries(
        raw=raw_data,
        sampling_rate=raw.info["sfreq"],
        domain_start=times[0],
        domain="auto",
    )

    # Events -> IrregularTimeSeries
    try:
        events, event_id = mne.events_from_annotations(raw)

        td_events = IrregularTimeSeries(
            timestamps=events[:, 0],
            event_code=events[:, -1],
            event_id=event_id,
        )
    except Exception:
        td_events = None

    out = Data(
        raw=td_raw,
    )

    raise NotImplementedError
