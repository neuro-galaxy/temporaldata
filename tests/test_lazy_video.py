"""LazyVideo tests. PyAV is the only video dependency the library uses."""

import copy
import os
import pickle
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("av")

from temporaldata import Data, Interval, LazyVideo
from temporaldata.lazy_video import _probe_segment


def _write_tiny_mp4(path: str, n_frames: int, w: int = 16, h: int = 16) -> None:
    """Encode a small zero-content H.264 mp4 as a test fixture."""
    import av

    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=5)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(n_frames):
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_h264_mp4_with_gop(
    path: str,
    n_frames: int,
    *,
    w: int = 64,
    h: int = 64,
    gop_size: int = 12,
    fps: int = 30,
) -> None:
    """Write a real H.264 mp4 with a known GOP structure and per-frame distinct
    colors so corruption is visually obvious in pixel comparisons.

    This is the fixture the mid-GOP regression test relies on -- without the
    fix, ``container.set(CAP_PROP_POS_FRAMES, k)`` for k != keyframe produced
    silently corrupted decodes (``mmco: unref short failure`` warnings).
    """
    import av

    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": str(gop_size), "keyint_min": str(gop_size)}

    rng = np.random.default_rng(seed=0)
    for i in range(n_frames):
        # A unique pattern per frame: frame index encoded in a smooth gradient
        # plus per-frame jitter. Smooth content compresses well so file size
        # stays small and decoded output is robust to encoder-side rounding.
        base = (i * 17) % 256
        img = np.full((h, w, 3), base, dtype=np.uint8)
        img[:, : w // 2, 1] = (base + 64) % 256
        img[: h // 2, :, 2] = (base + 128) % 256
        img += (rng.integers(0, 8, size=img.shape, dtype=np.int16)).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


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


def test_lazyvideo_cached_counts_skip_probe():
    """If segment_frame_counts is provided, __init__ must not re-probe."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 4
        _write_tiny_mp4(path, n)
        timestamps = np.linspace(0.0, 0.6, n, dtype=np.float64)

        with patch("temporaldata.lazy_video._probe_segment") as probe:
            probe.side_effect = AssertionError("packet probe must not be called")
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
            with patch("temporaldata.lazy_video._probe_segment") as probe:
                probe.side_effect = AssertionError("packet probe must not be called")
                loaded = LazyVideo.from_hdf5(f["video"])
                probe.assert_not_called()

        np.testing.assert_array_equal(
            loaded.segment_frame_counts, video.segment_frame_counts
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


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


def test_lazyvideo_concat_propagates_cached_segment_frame_counts():
    """LazyVideo.concat() must concatenate cached segment_frame_counts so the
    resulting object never has to re-probe. Regression for the case where
    torch_brain Dataset wraps cached single-segment videos.
    """
    paths = []
    try:
        # Build two single-segment LazyVideos with cached counts, then concat.
        videos = []
        for n in (3, 4):
            fd, path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            paths.append(path)
            _write_tiny_mp4(path, n)
            videos.append(
                LazyVideo(
                    timestamps=np.linspace(0.0, n * 0.2, n, dtype=np.float64),
                    video_file=path,
                    segment_frame_counts=np.array([n], dtype=np.int64),
                    channel_format="NHWC",
                )
            )

        with patch("temporaldata.lazy_video._probe_segment") as probe:
            probe.side_effect = AssertionError(
                "concat must not re-probe when inputs have cached counts"
            )
            merged = LazyVideo.concat(videos)
            probe.assert_not_called()

        np.testing.assert_array_equal(
            merged.segment_frame_counts, np.array([3, 4], dtype=np.int64)
        )
        assert merged.frame_count == 7
    finally:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)


def _decode_all_frames_av(path: str) -> np.ndarray:
    """Reference: decode every frame from start, presentation order, RGB.

    This is the "ground truth" for the mid-GOP regression test. Decoding
    sequentially from frame 0 has a complete reference chain so the output
    is always correct.
    """
    import av

    out = []
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            out.append(arr)
    finally:
        container.close()
    return np.stack(out, axis=0)


def _decode_all_frames_opencv_sequential(path: str) -> np.ndarray:
    """Pre-PyAV baseline: OpenCV sequential ``VideoCapture.read()``, RGB NHWC.

    This matches how older code obtained frames when reading the file from the
    beginning (reliable). Random access used ``CAP_PROP_POS_FRAMES`` and could
    return corrupted pixels mid-GOP; see ``test_lazyvideo_midgop_seek_*``.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    out: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(out, axis=0)


def test_lazyvideo_matches_opencv_sequential_decode():
    """LazyVideo (PyAV) must agree with the old OpenCV *sequential* read path.

    Before PyAV, stacks often used ``cv2.VideoCapture`` and pulled frames with
    repeated ``read()`` (or full linear scans). That output should match
    :meth:`LazyVideo._load_frames` for the same indices.

    This does **not** apply to ``VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, k)``
    followed by ``read()`` for arbitrary ``k``; that path disagreed with true
    decode mid-GOP and is why PyAV + keyframe-aligned seeks replaced it.
    """
    pytest.importorskip("cv2")

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        gop = 10
        n = 35
        _write_h264_mp4_with_gop(path, n, gop_size=gop, fps=30)

        opencv_rgb = _decode_all_frames_opencv_sequential(path)
        assert opencv_rgb.shape[0] == n, "OpenCV sequential frame count mismatch"

        timestamps = np.arange(n, dtype=np.float64) / 30.0
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            resize=None,
            colorspace="RGB",
            channel_format="NHWC",
        )

        # Full linear batch — same as scanning the file with read() in a loop.
        linear = video._load_frames(np.arange(n, dtype=np.int64))
        np.testing.assert_array_equal(
            linear,
            opencv_rgb,
            err_msg="LazyVideo must match OpenCV sequential decode (RGB NHWC)",
        )

        # Random order in one call — old stacks rarely did this via CAP_PROP;
        # still must index the same underlying RGB rows as OpenCV's ordered scan.
        indices = np.array([n - 1, 0, 11, 11, 7], dtype=np.int64)
        scrambled = video._load_frames(indices)
        for row, fi in enumerate(indices):
            np.testing.assert_array_equal(
                scrambled[row],
                opencv_rgb[fi],
                err_msg=f"frame index {fi} vs OpenCV reference",
            )
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_lazyvideo_midgop_seek_returns_correct_frames():
    """Regression: ``LazyVideo._load_frames`` must return frames bit-correct
    with sequential decode even when the slice starts mid-GOP.

    Before the PyAV port, ``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, k)`` for
    a non-keyframe ``k`` would log ``mmco: unref short failure`` and return
    silently corrupted frames (green blocks / ghosting) until the next IDR.
    This test slices windows that deliberately straddle GOP boundaries and
    asserts pixel-level agreement with the reference full-decode.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        gop = 12
        n = gop * 4 + 5  # 53 frames, several full GOPs + a tail
        _write_h264_mp4_with_gop(path, n, gop_size=gop, fps=30)

        reference = _decode_all_frames_av(path)
        assert reference.shape[0] == n

        timestamps = np.arange(n, dtype=np.float64) / 30.0
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            resize=None,
            colorspace="RGB",
            channel_format="NHWC",
        )

        # Pick a handful of mid-GOP starting indices: not on a keyframe.
        starts = [1, 5, 7, gop + 1, gop + 5, 2 * gop + 3, 3 * gop - 1]
        win_len = 6
        for s in starts:
            assert s % gop != 0, "test setup error: start must not be a keyframe"
            indices = np.arange(s, s + win_len, dtype=np.int64)
            frames = video._load_frames(indices)
            assert frames.shape == (win_len, *reference.shape[1:])
            # H.264 is lossy but deterministic for a given encoder/decoder
            # pair; the same decoded bytes should come out as long as the
            # reference chain is intact. The fix guarantees the chain is
            # intact, so frames must be exactly equal to the reference.
            np.testing.assert_array_equal(
                frames,
                reference[s : s + win_len],
                err_msg=f"mid-GOP slice starting at frame {s} differs from reference",
            )
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_lazyvideo_random_access_indices_pyav():
    """LazyVideo._load_frames with non-monotonic indices must still return
    each requested frame correctly placed in the output (sort-decode-unsort).
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        gop = 8
        n = 40
        _write_h264_mp4_with_gop(path, n, gop_size=gop, fps=30)
        reference = _decode_all_frames_av(path)

        timestamps = np.arange(n, dtype=np.float64) / 30.0
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            resize=None,
            colorspace="RGB",
            channel_format="NHWC",
        )

        # A deliberately scrambled order spanning multiple GOPs, including
        # going backwards (which forces a re-seek).
        indices = np.array([15, 3, 22, 8, 31, 17, 4, 39], dtype=np.int64)
        frames = video._load_frames(indices)
        assert frames.shape == (len(indices), *reference.shape[1:])
        for k, fi in enumerate(indices):
            np.testing.assert_array_equal(
                frames[k],
                reference[fi],
                err_msg=f"scrambled-index frame {fi} (output slot {k}) mismatch",
            )
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_lazyvideo_no_decoder_warnings_on_midgop_slice(caplog):
    """The fix's most user-visible promise: no libav warnings
    (``mmco: unref short failure`` etc.) when decoding mid-GOP slices.

    PyAV defaults to FFmpeg logging *off*; we explicitly enable it at VERBOSE
    here and route through Python's logging system so ``caplog`` can see any
    decoder complaints. With the keyframe-aligned seek + decode-forward fix,
    the H.264 decoder always has a complete reference chain and emits nothing.
    """
    import logging as _logging

    import av.logging as av_logging

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    prev_level = av_logging.get_level()
    av_logging.set_level(av_logging.VERBOSE)
    try:
        gop = 12
        n = gop * 3
        _write_h264_mp4_with_gop(path, n, gop_size=gop, fps=30)

        timestamps = np.arange(n, dtype=np.float64) / 30.0
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            channel_format="NHWC",
        )

        with caplog.at_level(_logging.DEBUG, logger="libav"):
            for s in (1, 5, gop + 1, gop + 7, 2 * gop + 3):
                video._load_frames(np.arange(s, s + 4, dtype=np.int64))

        offending = [
            r.getMessage()
            for r in caplog.records
            if "mmco" in r.getMessage() or "unref short" in r.getMessage()
        ]
        assert not offending, (
            f"libav decoder emitted MMCO warnings during mid-GOP slice: {offending}"
        )
    finally:
        av_logging.set_level(prev_level)
        if os.path.exists(path):
            os.remove(path)


def test_lazyvideo_pts_index_persists_through_hdf5(tmp_path):
    """Per-segment PTS arrays should round-trip via ``to_hdf5``/``from_hdf5``
    so a fresh process can decode without re-walking packets.
    """
    import h5py

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 24
        _write_h264_mp4_with_gop(path, n, gop_size=8, fps=30)
        timestamps = np.arange(n, dtype=np.float64) / 30.0
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            channel_format="NHWC",
        )
        # Force the PTS cache to populate.
        _ = video._load_frames(np.array([0, 1, 2], dtype=np.int64))
        assert video._pts_cache[0] is not None

        h5_path = tmp_path / "video.h5"
        with h5py.File(h5_path, "w") as f:
            video.to_hdf5(f.create_group("video"))

        with h5py.File(h5_path, "r") as f:
            assert "segment_pts_indices" in f["video"]
            with patch("temporaldata.lazy_video._probe_segment") as probe:
                probe.side_effect = AssertionError(
                    "PTS cache must be loaded from HDF5, not re-probed"
                )
                loaded = LazyVideo.from_hdf5(f["video"])
                # First decode shouldn't trigger _probe_segment either.
                loaded._load_frames(np.array([5, 10, 20], dtype=np.int64))
                probe.assert_not_called()

        np.testing.assert_array_equal(
            loaded._pts_cache[0], video._pts_cache[0]
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_probe_segment_pts_matches_decode_enumeration():
    """`_probe_segment` must list PTS in the same order as ``container.decode``.

    Regression: packet demux + sorted PTS can diverge from the decoder for some
    files; the probe uses the decode path so `_load_frames` targets the same
    presentation frames.
    """
    import av

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 40
        _write_h264_mp4_with_gop(path, n, gop_size=10, fps=30)
        count, pts_probe = _probe_segment(path)
        pts_dec: list[int] = []
        c = av.open(path)
        try:
            for f in c.decode(c.streams.video[0]):
                if f.pts is not None:
                    pts_dec.append(int(f.pts))
        finally:
            c.close()
        assert count == n == len(pts_dec)
        np.testing.assert_array_equal(pts_probe, np.asarray(pts_dec, dtype=np.int64))
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_lazyvideo_duplicate_frame_indices_in_one_batch():
    """Duplicate entries in `frame_indices` must not advance the decoder twice.

    Regression: with duplicate indices, `last_pts == target_pts` suppressed
    seeks but the iterator had already moved past the frame, yielding the wrong
    pixels for later duplicate slots (spurious decorrelation vs other modalities).
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        n = 20
        _write_h264_mp4_with_gop(path, n, gop_size=8, fps=30)
        reference = _decode_all_frames_av(path)
        timestamps = np.arange(n, dtype=np.float64) / 30.0
        video = LazyVideo(
            timestamps=timestamps,
            video_file=path,
            resize=None,
            colorspace="RGB",
            channel_format="NHWC",
        )
        indices = np.array([3, 3, 15, 15, 15, 7], dtype=np.int64)
        frames = video._load_frames(indices)
        for k, fi in enumerate(indices):
            np.testing.assert_array_equal(
                frames[k],
                reference[fi],
                err_msg=f"duplicate-index batch: slot {k} frame {fi}",
            )
    finally:
        if os.path.exists(path):
            os.remove(path)
