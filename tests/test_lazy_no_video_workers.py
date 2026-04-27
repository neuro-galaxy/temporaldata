"""Verification test: does multiprocessing DataLoader actually fail without
the Lazy* __reduce__ pickle hooks, when no LazyVideo is involved?

The claim under test: "I've used multiprocessing without a transform on
datasets without lazy video and only the other lazy properties, and it
worked."

This test stands up the leanest possible torch_brain pipeline:

  - synthetic brainset with ONLY h5py-backed lazy attrs (no LazyVideo)
  - num_workers > 0
  - no transform
  - the same `lambda b: b[0]` collate that countenance uses (does NOT touch
    sample attributes; just unwraps the batch list)

If this test passes WITHOUT the __reduce__ hooks, my mental model is wrong
and the pickling failure must be coming from elsewhere (probably
LazyVideo). If it fails WITHOUT the hooks and passes WITH them, then the
__reduce__ hooks really are necessary for any Lazy* type carried into a
worker.

Run procedure:
  1. With current code (__reduce__ hooks present): expect PASS.
  2. `git stash` the temporaldata/*.py edits and re-run: expect FAIL with
     "TypeError: h5py objects cannot be pickled" if my model is right.
  3. `git stash pop` to restore.
"""
from __future__ import annotations

import multiprocessing as mp

import h5py
import numpy as np
import pytest
import yaml

from temporaldata import (
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torch_brain.data.dataset")
pytest.importorskip("torch_brain.data.sampler")
pytest.importorskip("omegaconf")


def _build_session_no_video(session_id: str, brainset_id: str, *, duration: float = 30.0):
    n_irregular = 600
    fs_regular = 50.0
    n_regular = int(duration * fs_regular)

    irregular = IrregularTimeSeries(
        timestamps=np.linspace(0.0, duration, n_irregular).astype(np.float64),
        values=np.arange(n_irregular, dtype=np.float32),
        # Multiple keys so auto-promotion to non-lazy parent does NOT happen
        # after a single attribute access. This is critical to actually
        # exercise the bug.
        labels=np.array([f"v{i}" for i in range(n_irregular)]),
        more_data=np.arange(n_irregular, dtype=np.int64),
        domain="auto",
    )

    regular = RegularTimeSeries(
        signal=np.arange(n_regular * 4, dtype=np.float32).reshape(n_regular, 4),
        other_signal=np.arange(n_regular * 2, dtype=np.float32).reshape(n_regular, 2),
        sampling_rate=fs_regular,
        domain=Interval(0.0, duration - 1.0 / fs_regular),
    )

    intervals = Interval(
        start=np.array([0.0, 5.0, 12.0, 22.0]),
        end=np.array([4.5, 8.0, 18.0, 28.0]),
        label=np.array(["a", "b", "c", "d"]),
        score=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    return Data(
        brainset=Data(id=brainset_id),
        session=Data(id=session_id),
        subject=Data(id=f"{session_id}_subj"),
        irregular=irregular,
        regular=regular,
        intervals=intervals,
        domain=Interval(0.0, duration),
    )


@pytest.fixture
def no_video_brainset(tmp_path):
    brainset_id = "synthetic_nv"
    session_ids = ["sess_a", "sess_b"]

    bs_dir = tmp_path / brainset_id
    bs_dir.mkdir()

    for sid in session_ids:
        d = _build_session_no_video(sid, brainset_id, duration=30.0)
        with h5py.File(bs_dir / f"{sid}.h5", "w") as f:
            d.to_hdf5(f)

    config_path = tmp_path / "session.yaml"
    config_path.write_text(yaml.safe_dump(
        [{"selection": [{"brainset": brainset_id, "sessions": list(session_ids)}]}],
        sort_keys=False,
    ))
    return tmp_path, config_path, session_ids


def test_no_video_lazy_workers_actually_works(no_video_brainset):
    """The core question: with NO video and NO transform, does
    DataLoader(num_workers=2) survive shipping samples back?"""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method not supported on this platform")
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    from torch.utils.data import DataLoader
    from torch_brain.data.dataset import Dataset
    from torch_brain.data.sampler import SequentialFixedWindowSampler

    root, config_path, _ = no_video_brainset

    dataset = Dataset(root=str(root), config=str(config_path))
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals(),
        window_length=5.0,
        drop_short=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda b: b[0],
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
    )

    # The single critical assertion: iterating the loader does not raise.
    # If the pickle fix is absent, this will raise
    # "TypeError: h5py objects cannot be pickled" the moment the first
    # worker tries to put a sample on the result queue.
    samples = list(loader)
    assert len(samples) > 0

    dataset._close_open_files()
