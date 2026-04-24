"""LazyVideo tests (optional OpenCV + ffprobe)."""

import copy
import os
import pickle
import shutil
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("cv2")

from temporaldata import Data, Interval, LazyVideo


def _write_tiny_mp4(path: str, n_frames: int, w: int = 16, h: int = 16) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 5.0, (w, h))
    for _ in range(n_frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required for LazyVideo")
def test_lazyvideo_deepcopy_and_no_module_on_instance():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 3
        h, w = 16, 16
        _write_tiny_mp4(path, n, w=w, h=h)

        timestamps = np.linspace(0.0, 0.5, n, dtype=np.float64)
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            resize=None,
            colorspace="RGB",
            channel_format="NHWC",
        )

        assert not any(isinstance(v, types.ModuleType) for v in video.__dict__.values())

        data = Data(domain=Interval(0.0, 1.0), video=video)
        data_copy = copy.deepcopy(data)

        assert data_copy.video is not data.video
        assert not any(
            isinstance(v, types.ModuleType) for v in data_copy.video.__dict__.values()
        )
        frames = data_copy.video._load_frames(np.array([0], dtype=np.int64))
        assert frames.shape[0] == 1
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required for LazyVideo")
def test_lazyvideo_cached_counts_skip_ffprobe():
    """If segment_frame_counts is provided, __init__ must not call ffprobe."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 4
        _write_tiny_mp4(path, n)
        timestamps = np.linspace(0.0, 0.6, n, dtype=np.float64)

        with patch("temporaldata.lazy_video._probe_segment_frame_count") as probe:
            probe.side_effect = AssertionError("ffprobe must not be called")
            video = LazyVideo(
                timestamps=timestamps,
                video_file=path,
                segment_frame_counts=np.array([n], dtype=np.int64),
                channel_format="NHWC",
            )
            probe.assert_not_called()

        assert int(video.segment_frame_counts.sum()) == n
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required for LazyVideo")
def test_lazyvideo_is_picklable():
    """LazyVideo should be picklable (no live VideoCapture / module refs on the instance)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 3
        _write_tiny_mp4(path, n)
        timestamps = np.linspace(0.0, 0.4, n, dtype=np.float64)
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            segment_frame_counts=np.array([n], dtype=np.int64),
            channel_format="NHWC",
        )

        roundtrip = pickle.loads(pickle.dumps(video))
        assert roundtrip.video_files == video.video_files
        np.testing.assert_array_equal(
            roundtrip.segment_frame_counts, video.segment_frame_counts
        )
        # Reads should still work after unpickling (lazy-open).
        frames = roundtrip._load_frames(np.array([0, 1], dtype=np.int64))
        assert frames.shape[0] == 2
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required for LazyVideo")
def test_lazyvideo_to_from_hdf5_roundtrips_segment_counts(tmp_path):
    import h5py

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 3
        _write_tiny_mp4(path, n)
        timestamps = np.linspace(0.0, 0.4, n, dtype=np.float64)
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            channel_format="NHWC",
        )

        h5_path = tmp_path / "video.h5"
        with h5py.File(h5_path, "w") as f:
            video.to_hdf5(f.create_group("video"))

        with h5py.File(h5_path, "r") as f:
            assert "segment_frame_counts" in f["video"]
            with patch("temporaldata.lazy_video._probe_segment_frame_count") as probe:
                probe.side_effect = AssertionError("ffprobe must not be called")
                loaded = LazyVideo.from_hdf5(f["video"])
                probe.assert_not_called()

        np.testing.assert_array_equal(
            loaded.segment_frame_counts, video.segment_frame_counts
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required for LazyVideo")
def test_lazyvideo_slice_time_window_and_frames():
    """slice(start, end) keeps [start, end) in timestamp space and loads matching frames."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 5
        h, w = 16, 16
        _write_tiny_mp4(path, n, w=w, h=h)

        timestamps = np.linspace(0.0, 1.0, n, dtype=np.float64)
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            resize=None,
            colorspace="RGB",
            channel_format="NHWC",
        )

        sub = video.slice(0.2, 0.7)

        assert len(sub) == 2
        np.testing.assert_allclose(sub.timestamps, np.array([0.05, 0.30]))
        np.testing.assert_array_equal(sub.frame_indices, np.array([1, 2]))
        assert sub.frames.shape == (2, h, w, 3)
        assert sub.frames.dtype == np.uint8
    finally:
        if os.path.exists(path):
            os.remove(path)
