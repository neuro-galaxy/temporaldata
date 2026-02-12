"""
Interface for loading NWB files into temporaldata objects.

This module provides automatic conversion of NWB file contents to temporaldata
objects (Data, IrregularTimeSeries, RegularTimeSeries, Interval, ArrayDict).
"""

import importlib
import logging
from typing import Union, Dict, Any

import numpy as np

from .temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)


def _iterate_over_nwb(nwbfile, lazy_loading=True):
    """
    Iterate over NWB file objects and convert to temporaldata types.
    
    Parameters
    ----------
    nwbfile : pynwb.file.NWBFile
        NWB file object
    lazy_loading : bool
        If True, use lazy loading for data arrays
    
    Yields
    ------
    tuple
        (name, converted_object) where converted_object is a temporaldata object
    """
    pynwb = importlib.import_module("pynwb")
    
    for oid, obj in nwbfile.objects.items():
        obj_name = obj.name if hasattr(obj, "name") else f"object_{oid}"
        
        try:
            # Units table -> IrregularTimeSeries (spikes) + ArrayDict (units metadata)
            if isinstance(obj, pynwb.misc.Units):
                result = _make_units_table(obj, lazy_loading=lazy_loading)
                # Units table returns a dict with 'units' and 'spikes'
                for name, converted_obj in result.items():
                    if converted_obj is not None:
                        yield name, converted_obj
            
            # Time intervals (trials, epochs) -> Interval
            elif isinstance(obj, pynwb.epoch.TimeIntervals):
                converted_obj = _make_interval(obj)
                if converted_obj is not None:
                    yield obj_name, converted_obj
            
            # DynamicTable with timestamps -> could be events
            elif isinstance(obj, pynwb.misc.DynamicTable) and "timestamps" in obj.colnames:
                converted_obj = _make_timestamps(obj)
                if converted_obj is not None:
                    yield obj_name, converted_obj
            
            # TimeSeries -> Regular or Irregular based on timestamps
            elif isinstance(obj, pynwb.misc.TimeSeries):
                # Determine if regular or irregular
                if obj.timestamps is not None:
                    # Has explicit timestamps - treat as irregular
                    converted_obj = _make_irregular_timeseries(obj, lazy_loading=lazy_loading)
                else:
                    # Regular sampling rate
                    converted_obj = _make_regular_timeseries(obj, lazy_loading=lazy_loading)
                
                if converted_obj is not None:
                    yield obj_name, converted_obj
                    
        except Exception as e:
            logging.warning(f"Failed to convert {obj_name}: {e}")
            continue


def _make_interval(obj, **kwargs):
    """
    Convert NWB TimeIntervals to temporaldata Interval.
    
    Parameters
    ----------
    obj : pynwb.epoch.TimeIntervals
        NWB TimeIntervals object
        
    Returns
    -------
    Interval
        Converted interval object with metadata
    """
    if hasattr(obj, "to_dataframe"):
        df = obj.to_dataframe()
        
        # Rename standard columns
        if hasattr(df, "start_time") and hasattr(df, "stop_time"):
            df = df.rename(columns={"start_time": "start", "stop_time": "end"})
        
        # Create interval with all metadata
        if "start" in df.columns and "end" in df.columns:
            data = Interval.from_dataframe(df)
            return data
        else:
            logging.warning(f"TimeIntervals object missing start/end times, returning dataframe")
            return df
    
    return obj


def _make_irregular_timeseries(obj, lazy_loading=True, **kwargs):
    """
    Convert NWB TimeSeries to temporaldata IrregularTimeSeries.
    
    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB TimeSeries object
    lazy_loading : bool
        If True, return memory-view; if False, load data
        
    Returns
    -------
    IrregularTimeSeries
        Converted time series
    """
    d = obj.data
    t = obj.timestamps
    
    if not lazy_loading:
        t = t[:]
        d = d[:]
    
    # Determine domain
    domain = Interval(start=np.array([t[0]]), end=np.array([t[-1]]))
    
    # Create appropriate field name based on object type
    pynwb = importlib.import_module("pynwb")

    if isinstance(obj, pynwb.behavior.SpatialSeries):
        field_name = "position"
    else:
        field_name = "data"
    
    data = IrregularTimeSeries(
        timestamps=t,
        **{field_name: d},
        domain=domain,
    )
    
    return data


def _make_regular_timeseries(obj, lazy_loading=True, **kwargs):
    """
    Convert NWB TimeSeries to temporaldata RegularTimeSeries.
    
    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB TimeSeries object with regular sampling
    lazy_loading : bool
        If True, return memory-view; if False, load data
        
    Returns
    -------
    RegularTimeSeries
        Converted time series
    """
    d = obj.data
    if not lazy_loading:
        d = d[:]
    
    # Determine field name
    pynwb = importlib.import_module("pynwb")
    
    if isinstance(obj, pynwb.behavior.SpatialSeries):
        field_name = "position"
    else:
        field_name = "data"
    
    sampling_rate = obj.rate
    domain_start = obj.starting_time if obj.starting_time is not None else 0.0
    domain_end = domain_start + (obj.num_samples - 1) / sampling_rate
    
    data = RegularTimeSeries(
        **{field_name: d},
        sampling_rate=sampling_rate,
        domain=Interval(
            start=np.array([domain_start]),
            end=np.array([domain_end])
        ),
    )
    
    return data


def _make_units_table(obj, **kwargs):
    """
    Convert NWB Units table to ArrayDict + IrregularTimeSeries.
    
    Parameters
    ----------
    obj : pynwb.misc.Units
        NWB Units table
        
    Returns
    -------
    dict
        Dictionary with 'units' (ArrayDict) and 'spikes' (IrregularTimeSeries)
    """
    pynwb = importlib.import_module("pynwb")
    
    index = obj.id[:]
    
    # Extract spike times
    spike_timestamps = np.array([], dtype=np.float64)
    spike_unit_index = np.array([], dtype=np.uint32)
    
    for i, spike_train in zip(index, obj.spike_times_index[:]):
        spike_timestamps = np.concatenate([spike_timestamps, spike_train])
        spike_unit_index = np.concatenate(
            [spike_unit_index, np.full_like(spike_train, fill_value=i, dtype=np.uint32)]
        )
    
    # Create IrregularTimeSeries for spikes
    if len(spike_timestamps) > 0:
        spikes = IrregularTimeSeries(
            timestamps=spike_timestamps,
            unit_index=spike_unit_index,
            domain="auto",
        )
        spikes.sort()
    else:
        spikes = None
    
    # Extract metadata into ArrayDict
    metainfo = {"id": np.array(index)}
    
    units = ArrayDict(**metainfo)
    
    return {"units": units, "spikes": spikes}


def _make_timestamps(obj, **kwargs):
    """
    Convert NWB timestamp objects to IrregularTimeSeries.
    
    Parameters
    ----------
    obj : pynwb.misc.AnnotationSeries or pynwb.misc.DynamicTable
        NWB timestamp object
        
    Returns
    -------
    IrregularTimeSeries or dict
        Converted timestamps
    """
    if hasattr(obj, "timestamps"):
        timestamps = obj.timestamps[:]
        domain = Interval(start=np.array([timestamps[0]]), end=np.array([timestamps[-1]]))
        return IrregularTimeSeries(timestamps=timestamps, domain=domain)


def load_nwb(
    nwbfile: Any,
    lazy_loading: bool = True,
    auto_data: bool = True,
) -> Union[Data, Dict[str, Any]]:
    """
    Load an NWB file object and automatically convert to temporaldata objects.
    
    This function reads an NWB file object and converts its contents to appropriate
    temporaldata objects (IrregularTimeSeries, RegularTimeSeries, Interval, 
    ArrayDict). If auto_data=True, all objects are wrapped in a Data container.
    
    Parameters
    ----------
    nwbfile : pynwb.file.NWBFile
        An already-loaded NWB file object
    lazy_loading : bool, optional
        If True, data arrays are loaded lazily (memory-efficient).
        If False, all data is loaded into memory. Default is True.
    auto_data : bool, optional
        If True, wrap all loaded objects in a Data container with automatic
        domain inference. If False, return a dictionary of objects.
        Default is True.
        
    Returns
    -------
    Data or dict
        If auto_data=True, returns a Data object containing all converted objects.
        If auto_data=False, returns a dictionary mapping names to objects.
        
    Examples
    --------
    Load from an already-opened NWB file object:
    
    >>> from pynwb import NWBHDF5IO
    >>> from temporaldata.io import load_nwb
    >>> 
    >>> io = NWBHDF5IO("recording.nwb", "r")
    >>> nwbfile = io.read()
    >>> data = load_nwb(nwbfile)
    >>> 
    >>> # Access converted data
    >>> data.spikes  # IrregularTimeSeries with spike times
    >>> data.units   # ArrayDict with unit metadata
    >>> data.trials  # Interval with trial information
    >>> 
    >>> io.close()
    
    Load without automatic Data wrapping:
    
    >>> io = NWBHDF5IO("recording.nwb", "r")
    >>> nwbfile = io.read()
    >>> objects = load_nwb(nwbfile, auto_data=False)
    >>> spikes = objects["spikes"]
    >>> units = objects["units"]
    >>> io.close()
    
    Notes
    -----
    The function automatically detects and converts:
    - Units tables (pynwb.misc.Units) -> IrregularTimeSeries (spikes) + ArrayDict (metadata)
    - TimeIntervals (pynwb.epoch.TimeIntervals) -> Interval (trials, epochs)
    - TimeSeries (pynwb.misc.TimeSeries) -> RegularTimeSeries or IrregularTimeSeries
    - Behavioral data -> RegularTimeSeries or IrregularTimeSeries
    """
    # Extract and convert all compatible objects
    # _iterate_over_nwb returns (name, converted_object) tuples
    objects = dict(_iterate_over_nwb(nwbfile, lazy_loading=lazy_loading))
    
    if auto_data:
        data = Data(**objects, domain="auto")
        return data
    else:
        return objects
