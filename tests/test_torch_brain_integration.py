"""End-to-end integration test that mirrors the real countenance/build_dataset.py
failure mode.

This test stands up a synthetic brainset on disk in the exact layout
``torch_brain.data.dataset.Dataset`` expects, configures a YAML selection,
and runs the full pipeline:

    Dataset(root, config) -> SequentialFixedWindowSampler -> DataLoader(num_workers>0)

The DataLoader workers slice ``Data`` objects whose children are Lazy*
variants. Pre-fix this raised ``TypeError: h5py objects cannot be pickled``
when the worker tried to ship the sample back to the main process. The
``__reduce__`` hook on each ``Lazy*`` class added in this branch should
make every sample materialize automatically before pickle, and this test
proves it end-to-end.

The test is skipped when torch / torch_brain / omegaconf aren't installed
in the test env so the broader temporaldata suite stays lightweight.
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
    LazyInterval,
    LazyIrregularTimeSeries,
    LazyRegularTimeSeries,
    LazyVideo,
    RegularTimeSeries,
)


# ---------------------------------------------------------------------------
# environment guards: skip cleanly if heavy deps aren't installed
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
torch_brain_data = pytest.importorskip("torch_brain.data.dataset")
torch_brain_sampler = pytest.importorskip("torch_brain.data.sampler")
pytest.importorskip("omegaconf")
pytest.importorskip("av")  # required for LazyVideo

# This integration test exists to validate the picklability fix added in
# temporaldata's Lazy* __reduce__ hooks. If the temporaldata loaded by the
# test environment doesn't carry that fix, the test would either fail with
# the old "h5py objects cannot be pickled" error or pass for the wrong
# reasons. Fail loudly instead so the env gets fixed.
import temporaldata.irregular_ts as _itts

if not hasattr(_itts.LazyIrregularTimeSeries, "__reduce__") or (
    _itts.LazyIrregularTimeSeries.__reduce__ is object.__reduce__
):
    pytest.skip(
        "loaded temporaldata is missing Lazy* __reduce__ pickle hooks; "
        f"using {_itts.__file__} -- install the local editable build",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# fixtures: build a synthetic brainset that torch_brain.Dataset can ingest
# ---------------------------------------------------------------------------


_VIDEO_FPS = 5.0
_VIDEO_W = 16
_VIDEO_H = 16


def _write_tiny_mp4(path, n_frames: int, w: int = _VIDEO_W, h: int = _VIDEO_H) -> None:
    """Write a tiny mp4 with ``n_frames`` distinguishable solid-color frames
    so we can sanity-check that the worker actually decoded the right
    range."""
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=int(_VIDEO_FPS))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        # Vary intensity per frame so different windows produce different
        # numerical fingerprints; useful for downstream sanity assertions.
        img = np.full((h, w, 3), (i * 5) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _build_session_data(
    session_id: str,
    brainset_id: str,
    video_path,
    *,
    duration: float = 30.0,
):
    """Construct a Data object with the structure torch_brain expects:
    nested ``brainset``, ``session``, and ``subject`` Data containers (each
    with an ``id``), plus several time-based attributes that will become
    Lazy* on read. Includes a ``LazyVideo`` backed by ``video_path`` so the
    test exercises the same code path countenance/build_dataset.py uses.

    Duration is in seconds. Video must already exist at ``video_path``."""
    n_irregular = 600  # 20 Hz over 30s
    fs_regular = 50.0
    n_regular = int(duration * fs_regular)
    n_video_frames = int(duration * _VIDEO_FPS)

    irregular = IrregularTimeSeries(
        timestamps=np.linspace(0.0, duration, n_irregular).astype(np.float64),
        values=np.arange(n_irregular, dtype=np.float32),
        labels=np.array([f"v{i}" for i in range(n_irregular)]),
        domain="auto",
    )

    regular = RegularTimeSeries(
        signal=np.arange(n_regular * 4, dtype=np.float32).reshape(n_regular, 4),
        sampling_rate=fs_regular,
        domain=Interval(0.0, duration - 1.0 / fs_regular),
    )

    intervals = Interval(
        start=np.array([0.0, 5.0, 12.0, 22.0]),
        end=np.array([4.5, 8.0, 18.0, 28.0]),
        label=np.array(["a", "b", "c", "d"]),
    )

    # Pass segment_frame_counts so __init__ skips the (slow, optional)
    # ffprobe pass -- we already know the count because we just wrote the
    # mp4. This keeps the test independent of whether ffprobe is on $PATH.
    video_timestamps = np.arange(n_video_frames, dtype=np.float64) / _VIDEO_FPS
    video = LazyVideo(
        timestamps=video_timestamps,
        video_file=str(video_path),
        resize=None,
        colorspace="RGB",
        channel_format="NHWC",
        segment_frame_counts=np.array([n_video_frames], dtype=np.int64),
    )

    return Data(
        brainset=Data(id=brainset_id),
        session=Data(id=session_id),
        subject=Data(id=f"{session_id}_subj"),
        irregular=irregular,
        regular=regular,
        intervals=intervals,
        video=video,
        domain=Interval(0.0, duration),
    )


@pytest.fixture
def synthetic_brainset(tmp_path):
    """Lay out a brainset on disk and return (root_dir, config_path,
    expected_sessions). Layout matches what torch_brain.Dataset.config
    expects:

        <root>/<brainset_id>/<session_id>.h5
    """
    brainset_id = "synthetic"
    session_ids = ["sess_a", "sess_b"]

    bs_dir = tmp_path / brainset_id
    bs_dir.mkdir()

    for sid in session_ids:
        video_path = bs_dir / f"{sid}.mp4"
        # 5 fps * 30s = 150 video frames, plenty to cover any 5s window.
        _write_tiny_mp4(video_path, n_frames=int(30.0 * _VIDEO_FPS))

        d = _build_session_data(sid, brainset_id, video_path, duration=30.0)
        with h5py.File(bs_dir / f"{sid}.h5", "w") as f:
            d.to_hdf5(f)

    config_path = tmp_path / "session.yaml"
    config = [
        {
            "selection": [
                {"brainset": brainset_id, "sessions": list(session_ids)}
            ]
        }
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return tmp_path, config_path, session_ids


# ---------------------------------------------------------------------------
# the actual test: real torch_brain Dataset + DataLoader + workers
# ---------------------------------------------------------------------------


def _is_lazy_variant(obj) -> bool:
    return isinstance(
        obj, (LazyIrregularTimeSeries, LazyRegularTimeSeries, LazyInterval)
    )


def _has_h5py_refs(obj, _seen=None) -> bool:
    """Walk a sample and return True if any attribute is still a live
    h5py.Dataset / Group / File. Used as a strong negative assertion --
    if any ref survived to the main process, the picklability fix
    regressed."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return False
    _seen.add(id(obj))

    if isinstance(obj, (h5py.Dataset, h5py.Group, h5py.File)):
        return True
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            if _has_h5py_refs(v, _seen):
                return True
    return False


@pytest.mark.parametrize("num_workers", [2])
def test_torch_brain_dataloader_with_workers(synthetic_brainset, num_workers):
    """Full pipeline test: synthetic brainset -> torch_brain.Dataset ->
    SequentialFixedWindowSampler -> DataLoader(num_workers=N).

    Pre-fix this raised ``TypeError: h5py objects cannot be pickled`` at
    the worker -> main queue handoff. The ``__reduce__`` hooks on each
    Lazy* class force materialization in the worker so the sample is
    pickle-safe by the time it leaves the process.

    We force ``fork`` start method because that's what the user's Linux
    setup uses by default (and is where the original bug was hit) and
    because ``torch_brain.data.dataset`` defines its default prefix
    functions as module-level lambdas, which the ``spawn`` start method
    cannot pickle when shipping the dataset to a worker. Fork reproduces
    the actual h5py-pickling failure without that unrelated obstacle."""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method not supported on this platform")
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    from torch.utils.data import DataLoader
    from torch_brain.data.dataset import Dataset
    from torch_brain.data.sampler import SequentialFixedWindowSampler

    root, config_path, session_ids = synthetic_brainset

    dataset = Dataset(root=str(root), config=str(config_path))
    sampling_intervals = dataset.get_sampling_intervals()

    # 5s windows over 30s sessions -> 6 windows per session, 12 total.
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=5.0,
        drop_short=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda b: b[0],  # mirror countenance/build_dataset.py
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    samples = list(loader)

    # Sanity: the sampler should produce some windows.
    assert len(samples) > 0, "sampler produced no windows"

    for sample in samples:
        # 1) The sample must NOT carry live h5py refs. This is the strong
        #    assertion that the picklability fix actually materialized
        #    everything in the worker.
        assert not _has_h5py_refs(sample), (
            "sample still contains live h5py refs after crossing "
            "the multiprocessing boundary; the Lazy* __reduce__ "
            "materialization regressed"
        )

        # 2) The sample's lazy children should have been promoted to
        #    their non-lazy parent classes by load() during pickling.
        for attr in ("irregular", "regular", "intervals"):
            child = getattr(sample, attr)
            assert not _is_lazy_variant(child), (
                f"{attr} is still a Lazy* after pickle round-trip; "
                f"got {type(child).__name__}"
            )

        # 3) Numeric content should be readable and non-empty for windows
        #    that actually overlap data.
        ts = sample.irregular.timestamps
        vals = sample.irregular.values
        assert isinstance(ts, np.ndarray)
        assert isinstance(vals, np.ndarray)
        assert ts.shape[0] == vals.shape[0]

        sig = sample.regular.signal
        assert isinstance(sig, np.ndarray)
        assert sig.ndim == 2 and sig.shape[1] == 4

        # 4) Video: Data.slice promotes LazyVideo -> IrregularTimeSeries
        #    with a fully-decoded `.frames` array (no h5py / no live decoder).
        #    This is the same code path countenance/build_dataset.py uses.
        video = sample.video
        assert not isinstance(video, LazyVideo), (
            f"video is still a LazyVideo after slice; got {type(video).__name__}"
        )
        assert isinstance(video.frames, np.ndarray)
        assert isinstance(video.timestamps, np.ndarray)
        # 5s window @ 5fps -> ~25 frames. Allow some slack for boundary
        # alignment; a non-empty result is the meaningful check.
        assert video.frames.shape[0] > 0
        assert video.frames.shape[0] == video.timestamps.shape[0]
        # NHWC layout from _build_session_data; H/W must match the mp4
        # we wrote, channels = 3 for RGB.
        assert video.frames.shape[1:] == (_VIDEO_H, _VIDEO_W, 3)
        assert video.frames.dtype == np.uint8

        # 5) Identifiers survived (Data.brainset.id / Data.session.id).
        assert sample.brainset.id == "synthetic"
        # session.id is mutated by torch_brain to "<brainset>/<session>".
        assert sample.session.id.startswith("synthetic/sess_")

    # Cleanup: close the dataset's open file handles.
    dataset._close_open_files()


def test_torch_brain_dataloader_zero_workers_baseline(synthetic_brainset):
    """Baseline: same pipeline with num_workers=0 (no pickling). Confirms
    that the test setup itself is sound -- if this fails, the synthetic
    brainset is wrong, not the picklability fix."""
    from torch.utils.data import DataLoader
    from torch_brain.data.dataset import Dataset
    from torch_brain.data.sampler import SequentialFixedWindowSampler

    root, config_path, _session_ids = synthetic_brainset

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
        num_workers=0,
    )

    samples = list(loader)
    assert len(samples) > 0
    # With workers=0 the lazy types may STILL be lazy (no pickle to force
    # materialization). Touching an attribute should resolve them on the fly.
    s = samples[0]
    _ = s.irregular.timestamps  # must not raise
    _ = s.regular.signal
    # LazyVideo.slice() always eagerly decodes frames, so video.frames
    # must already be a numpy array regardless of worker count.
    assert isinstance(s.video.frames, np.ndarray)
    assert s.video.frames.shape[0] > 0

    dataset._close_open_files()
