"""Benchmark suite for temporaldata hot paths.

Usage:
    uv run python benchmarks/bench_hotpath.py [--save FILENAME]

When --save is provided, results are appended as a JSON record to the file.
"""

from __future__ import annotations

import argparse
import json
import time
import timeit
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    LazyInterval,
    RegularTimeSeries,
)


def _bench(label: str, stmt, number: int, setup=lambda: None) -> dict:
    setup()
    times = timeit.repeat(stmt, number=number, repeat=5)
    mean_us = min(times) / number * 1e6
    return {"label": label, "number": number, "mean_us": round(mean_us, 3)}


def bench_arraydict_keys():
    ad = ArrayDict(**{f"key_{i}": np.arange(100, dtype=np.float64) for i in range(10)})

    def go():
        ad.keys()

    return _bench("ArrayDict.keys() x100k", go, number=100_000)


def _make_disjoint_intervals(
    n, span=10_000, min_gap=1.0, min_dur=0.5, max_dur=2.0, seed=42
):
    """Generate n sorted disjoint intervals spread over [0, span]."""
    rng = np.random.default_rng(seed)
    starts = np.empty(n, dtype=np.float64)
    ends = np.empty(n, dtype=np.float64)
    t = 0.0
    for i in range(n):
        t += rng.uniform(min_gap, min_gap + 3.0)
        dur = rng.uniform(min_dur, max_dur)
        starts[i] = t
        ends[i] = t + dur
        t = ends[i]
    return Interval(start=starts, end=ends)


def bench_interval_and_single():
    d1 = _make_disjoint_intervals(1000, seed=42)
    single = Interval(500.0, 600.0)

    def go():
        d1 & single

    return _bench("Interval.__and__ (single)", go, number=1_000)


def bench_interval_and_multi():
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1 & d2

    return _bench("Interval.__and__ (multi)", go, number=200)


def bench_interval_or():
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1 | d2

    return _bench("Interval.__or__", go, number=200)


def bench_interval_difference():
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1.difference(d2)

    return _bench("Interval.difference", go, number=200)


def bench_data_slice():
    rng = np.random.default_rng(42)
    n_spikes = 50_000
    ts = np.sort(rng.uniform(0, 1000, n_spikes))
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=ts,
            unit_index=rng.integers(0, 100, n_spikes),
            domain=Interval(0.0, 1000.0),
        ),
        lfp=RegularTimeSeries(
            raw=rng.standard_normal((250_000, 4)),
            sampling_rate=250.0,
            domain=Interval(0.0, 1000.0),
        ),
        trials=Interval(
            start=np.arange(0, 1000, 10, dtype=np.float64),
            end=np.arange(10, 1010, 10, dtype=np.float64),
        ),
        domain=Interval(
            start=np.array([0.0, 500.0]),
            end=np.array([400.0, 1000.0]),
        ),
    )

    def go():
        data.slice(100.0, 200.0)

    return _bench("Data.slice() end-to-end", go, number=500)


def bench_lazy_interval_access():
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    path = tmpfile.name
    tmpfile.close()

    rng = np.random.default_rng(42)
    starts = np.sort(rng.uniform(0, 10_000, 500))
    ends = starts + rng.uniform(0.5, 2.0, 500)
    iv = Interval(start=starts, end=ends)

    with h5py.File(path, "w") as f:
        iv.to_hdf5(f)

    results = []
    with h5py.File(path, "r") as f:

        def go():
            lazy = LazyInterval.from_hdf5(f)
            _ = lazy.start
            _ = lazy.end

        results = _bench("LazyInterval .start/.end access", go, number=2_000)

    os.unlink(path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    benchmarks = [
        bench_arraydict_keys,
        bench_interval_and_single,
        bench_interval_and_multi,
        bench_interval_or,
        bench_interval_difference,
        bench_data_slice,
        bench_lazy_interval_access,
    ]

    results = []
    print(f"{'Benchmark':<42} {'Iters':>8} {'Mean (µs)':>12}")
    print("-" * 65)
    for bench_fn in benchmarks:
        r = bench_fn()
        results.append(r)
        print(f"{r['label']:<42} {r['number']:>8} {r['mean_us']:>12.3f}")

    if args.save:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        with open(args.save, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
