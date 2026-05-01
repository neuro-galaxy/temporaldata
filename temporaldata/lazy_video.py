from __future__ import annotations

import logging
from typing import Sequence

import h5py
import numpy as np

from .irregular_ts import IrregularTimeSeries

_av = None


def _av_module():
    """Lazy-import PyAV once; do not store the module on LazyVideo instances."""
    global _av
    if _av is None:
        try:
            import av as _av_mod
        except ImportError as exc:  # pragma: no cover - exercised only without PyAV
            raise ImportError(
                "PyAV not installed, you must install temporaldata using "
                "`pip install -e .[video]`"
            ) from exc
        _av = _av_mod
    return _av


def _probe_segment_decode(path: str) -> np.ndarray:
    """Walk ``container.decode()`` to collect each frame's PTS in presentation
    order. Slow (full IDCT + motion comp for every frame) but matches what
    :meth:`LazyVideo._load_frames` actually iterates."""
    av = _av_module()
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        ptses: list[int] = []
        for frame in container.decode(stream):
            if frame.pts is not None:
                ptses.append(int(frame.pts))
        return np.asarray(ptses, dtype=np.int64)
    finally:
        container.close()


def _probe_segment_demux(path: str) -> np.ndarray:
    """Walk ``container.demux()`` and collect packet PTS, sorted ascending.

    Order-of-magnitude faster than :func:`_probe_segment_decode` because it
    skips the actual decode pipeline (no IDCT, no motion comp, no swscale).

    Caveat: this assumes ``sorted(packet.pts) == decode_order_pts`` for the
    target file. This holds for the constellation H.264-in-MKV recordings
    we've measured (verified across P_053 and P_055), and for any standard
    GOP-structured H.264/H.265 stream. :func:`_probe_segment` falls back to
    the decode path if validation fails.
    """
    av = _av_module()
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        ptses: list[int] = []
        for packet in container.demux(stream):
            if packet.pts is not None:
                ptses.append(int(packet.pts))
        ptses.sort()
        return np.asarray(ptses, dtype=np.int64)
    finally:
        container.close()


def _probe_segment(path: str) -> tuple[int, np.ndarray]:
    """Return ``(frame_count, pts_array)`` for ``path``.

    Tries the fast demux-only path first (~10–50× faster than full decode on
    the external SSD). Returns whatever it produced; the caller (typically
    :meth:`LazyVideo._ensure_pts_table`) cross-checks the length against the
    expected ``segment_frame_counts`` and tolerates a small overshoot. If the
    fast path hits an exception (corrupt container, unusual codec, etc.) we
    fall back to the slow decode-order probe.
    """
    try:
        ptses_arr = _probe_segment_demux(path)
        if ptses_arr.size > 0:
            return len(ptses_arr), ptses_arr
    except Exception as ex:  # pragma: no cover - exercised on malformed files only
        logging.warning(
            "LazyVideo: demux probe failed for %s (%s); falling back to "
            "decode probe.",
            path,
            ex,
        )
    ptses_arr = _probe_segment_decode(path)
    return len(ptses_arr), ptses_arr


class LazyVideo(object):
    r"""An object that lazily loads batches of video data using PyAV.

    Frames are decoded on demand inside :meth:`slice` / :meth:`_load_frames`.
    Seeks are keyframe-aligned (``container.seek(pts, any_frame=False,
    backward=True)``) and the decoder is then advanced frame-by-frame to the
    requested PTS, so frames returned mid-GOP are bit-correct (no
    ``mmco: unref short failure`` corruption).

    Args:
        timestamps: array of camera timestamps, one per frame in presentation
            order.
        video_file: absolute path to a single video file, or a sequence of
            paths to be treated as concatenated segments.
        resize: ``(height, width)`` to resize frames to, or ``None`` to keep
            original dimensions.
        colorspace: ``"RGB"`` or ``"G"``.
        channel_format: ``"NCHW"`` or ``"NHWC"``.
        segment_frame_counts: optional cached per-segment frame counts. When
            absent, PyAV is used to demux packets and count them.
        segment_pts_indices: optional cached per-segment PTS arrays. Each entry
            is a presentation-ordered ``int64`` ndarray of length matching the
            corresponding segment's frame count (``None`` is allowed for
            individual entries; missing ones are demuxed lazily on first use).
            Caching these avoids a second packet-walk per segment per process.
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        video_file: str | Sequence[str],
        resize: tuple | None = None,
        colorspace: str = "RGB",
        channel_format: str = "NCHW",
        segment_frame_counts: np.ndarray | Sequence[int] | None = None,
        segment_pts_indices: Sequence[np.ndarray | None] | None = None,
    ):
        # Validate PyAV is importable but do not actually open any container
        # here. Containers are opened per-slice in `_load_frames` so that:
        #   * objects are picklable (multi-worker DataLoader-friendly),
        #   * file descriptors don't pile up across many sessions,
        #   * decoder state is fresh for each window.
        _av_module()

        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        if isinstance(video_file, str):
            video_files = [video_file]
        else:
            video_files = [str(path) for path in video_file]
        if len(video_files) == 0:
            raise ValueError("At least one video file must be provided.")
        self.video_files = video_files
        self.video_file = (
            video_files[0] if len(video_files) == 1 else list(video_files)
        )

        if (resize is None) or (isinstance(resize, tuple) and len(resize) == 2):
            self.resize = resize
        else:
            raise ValueError('"resize" arg must be None or a tuple (height, width)')

        if colorspace not in ("RGB", "G"):
            raise ValueError('"colorspace" arg must be "RGB" or "G"')
        self.colorspace = colorspace

        if channel_format not in ("NCHW", "NHWC"):
            raise ValueError('"channel_format" arg must be "NCHW" or "NHWC"')
        self.channel_format = channel_format

        # Resolve segment_frame_counts and the per-segment PTS cache. There are
        # three cases:
        #   1. Both supplied: validate consistency, accept both.
        #   2. Counts supplied, PTS not: build PTS lazily on first decode.
        #   3. Neither supplied: walk every segment with PyAV (slow path).
        n_seg = len(video_files)

        if segment_pts_indices is not None:
            raw_pts = list(segment_pts_indices)
            if len(raw_pts) != n_seg:
                raise ValueError(
                    f"segment_pts_indices has length {len(raw_pts)} but there "
                    f"are {n_seg} video files."
                )
            pts_cache: list[np.ndarray | None] = [
                None if p is None else np.asarray(p, dtype=np.int64) for p in raw_pts
            ]
            if all(p is not None for p in pts_cache):
                counts_from_pts = np.array(
                    [len(p) for p in pts_cache], dtype=np.int64
                )
                if segment_frame_counts is None:
                    segment_frame_counts = counts_from_pts
                else:
                    counts_arr = np.asarray(
                        list(segment_frame_counts), dtype=np.int64
                    )
                    if not np.array_equal(counts_arr, counts_from_pts):
                        raise ValueError(
                            "segment_frame_counts disagrees with the lengths "
                            "of segment_pts_indices."
                        )
                    segment_frame_counts = counts_arr
            else:
                if segment_frame_counts is None:
                    raise ValueError(
                        "segment_pts_indices contains None entries; "
                        "segment_frame_counts must also be provided."
                    )
        elif segment_frame_counts is not None:
            pts_cache = [None] * n_seg
        else:
            print(
                f"[WARNING] cache miss for segment frame counts for "
                f"{self.video_files}"
            )
            counts: list[int] = []
            pts_cache = []
            for path in self.video_files:
                count, pts_arr = _probe_segment(path)
                counts.append(count)
                pts_cache.append(pts_arr)
            segment_frame_counts = np.asarray(counts, dtype=np.int64)

        segment_frame_counts = np.asarray(
            list(segment_frame_counts), dtype=np.int64
        )
        if len(segment_frame_counts) != n_seg:
            raise ValueError(
                f"segment_frame_counts has length {len(segment_frame_counts)} "
                f"but there are {n_seg} video files."
            )

        self.segment_frame_counts = segment_frame_counts
        self.segment_frame_offsets = np.cumsum(
            np.concatenate(([0], self.segment_frame_counts[:-1]))
        )
        self._pts_cache = pts_cache

        frame_count = int(self.segment_frame_counts.sum())
        if frame_count != self.timestamps.shape[0]:
            if frame_count > self.timestamps.shape[0]:
                # If the video has trailing frames past the timestamp series we
                # can safely ignore them.
                frame_count = self.timestamps.shape[0]
            else:
                raise ValueError(
                    f"Frame count mismatch: {frame_count} != "
                    f"{self.timestamps.shape[0]}"
                )
        self.frame_count = frame_count

        self.frame_indices = np.arange(frame_count, dtype=np.int64)

        if frame_count > 1 and np.any(np.diff(self.timestamps[:frame_count]) < 0):
            sort_idx = np.argsort(self.timestamps[:frame_count])
            self.timestamps[:frame_count] = self.timestamps[sort_idx]
            self.frame_indices = sort_idx.astype(np.int64)
            logging.info(
                "LazyVideo: sorted %d timestamps that were not monotonically "
                "increasing",
                frame_count,
            )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        vf = self.video_files[0] if len(self.video_files) == 1 else list(self.video_files)
        # Carry the PTS cache through so the copy doesn't re-walk packets on
        # first decode. Pass-through of `None` entries is supported.
        pts_arg: list[np.ndarray | None] | None
        if any(p is not None for p in self._pts_cache):
            pts_arg = [None if p is None else p.copy() for p in self._pts_cache]
        else:
            pts_arg = None
        dup = self.__class__(
            timestamps=self.timestamps.copy(),
            video_file=vf,
            resize=self.resize,
            colorspace=self.colorspace,
            channel_format=self.channel_format,
            segment_frame_counts=self.segment_frame_counts.copy(),
            segment_pts_indices=pts_arg,
        )
        memo[id(self)] = dup
        return dup

    def __len__(self):
        r"""Returns the first dimension of timestamps."""
        return self.frame_count

    def __repr__(self):
        cls = self.__class__.__name__
        info = ",\n".join(
            [
                f"timestamps=[{self.frame_count}]",
                f"frames=[{self.frame_count}]",
                f"segments=[{len(self.video_files)}]",
            ]
        )
        return f"{cls}(\n{info}\n)"

    @classmethod
    def concat(cls, videos: Sequence["LazyVideo"]) -> "LazyVideo":
        if len(videos) == 0:
            raise ValueError("Expected at least one LazyVideo.")
        if any(not isinstance(video, cls) for video in videos):
            raise ValueError("All objects must be LazyVideo instances.")

        first = videos[0]
        for video in videos[1:]:
            if video.resize != first.resize:
                raise ValueError("All LazyVideo objects must share the same resize.")
            if video.colorspace != first.colorspace:
                raise ValueError("All LazyVideo objects must share the same colorspace.")
            if video.channel_format != first.channel_format:
                raise ValueError(
                    "All LazyVideo objects must share the same channel_format."
                )

        timestamps = np.concatenate([video.timestamps for video in videos], axis=0)
        video_files = [
            video_file for video in videos for video_file in video.video_files
        ]
        segment_frame_counts = np.concatenate(
            [np.asarray(video.segment_frame_counts, dtype=np.int64) for video in videos],
            axis=0,
        )
        # Propagate PTS caches so the merged video doesn't re-demux any segment
        # that was already cached on its source.
        merged_pts: list[np.ndarray | None] = []
        for video in videos:
            merged_pts.extend(video._pts_cache)
        pts_arg = merged_pts if any(p is not None for p in merged_pts) else None
        return cls(
            timestamps=timestamps,
            video_file=video_files,
            resize=first.resize,
            colorspace=first.colorspace,
            channel_format=first.channel_format,
            segment_frame_counts=segment_frame_counts,
            segment_pts_indices=pts_arg,
        )

    def _segment_for_frame(self, frame_index: int):
        segment_idx = int(
            np.searchsorted(self.segment_frame_offsets, frame_index, side="right") - 1
        )
        segment_start = int(self.segment_frame_offsets[segment_idx])
        local_index = int(frame_index - segment_start)
        return segment_idx, local_index

    def _ensure_pts_table(self, segment_idx: int) -> np.ndarray:
        cached = self._pts_cache[segment_idx]
        if cached is not None:
            return cached
        _, pts_arr = _probe_segment(self.video_files[segment_idx])
        expected = int(self.segment_frame_counts[segment_idx])
        if len(pts_arr) != expected:
            # Tolerate a small overshoot (we already do this for total frame
            # counts in __init__): the LazyVideo only addresses the first
            # `expected` frames in presentation order.
            if len(pts_arr) > expected:
                pts_arr = pts_arr[:expected]
            else:
                raise ValueError(
                    f"Segment {segment_idx} ({self.video_files[segment_idx]}) "
                    f"reports {expected} frames in segment_frame_counts but "
                    f"PyAV demuxed only {len(pts_arr)} packets with PTS."
                )
        self._pts_cache[segment_idx] = pts_arr
        return pts_arr

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`IrregularTimeSeries` object that contains the data
        between the start and end times. The end time is exclusive, the slice will
        only include data in :math:`[\textrm{start}, \textrm{end})`.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, the returned ``timestamps`` are shifted
                to be relative to ``start`` (matching
                :meth:`IrregularTimeSeries.slice`'s default). If :obj:`False`,
                ``timestamps`` are returned in the same time base as
                ``self.timestamps`` (absolute camera time). Defaults to
                :obj:`True`. ``Data.slice`` forwards its own ``reset_origin`` here
                so video timestamps stay aligned with the rest of the sliced
                data, instead of silently resetting and breaking
                timestamp-based joins.
        """
        timestamps = IrregularTimeSeries(
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            frame_indices=self.frame_indices,
            domain="auto",
        )
        timestamps_sliced = timestamps.slice(
            start=start, end=end, reset_origin=reset_origin
        )
        frames_sliced = self._load_frames(timestamps_sliced.frame_indices)
        timestamps_sliced.frames = frames_sliced
        return timestamps_sliced

    def _empty_frames_array(self) -> np.ndarray:
        n_channels = 3 if self.colorspace == "RGB" else 1
        if self.channel_format == "NCHW":
            return np.zeros((0, n_channels, 1, 1), dtype="uint8")
        return np.zeros((0, 1, 1, n_channels), dtype="uint8")

    def _load_frames(self, frame_indices: np.ndarray):
        """Decode the requested presentation-ordered frames.

        Implementation notes (speed):
          * Indices are sorted by ``(segment, local_index)`` so each segment is
            walked forward in presentation order; the only seeks issued are at
            segment boundaries (or when the caller passes a non-monotonic
            sequence). For a typical 10s window this is one seek per slice.
          * Decoding runs **single-threaded** (``stream.thread_count = 1``,
            ``stream.thread_type = "NONE"``). FFmpeg's frame/slice threading is
            ~2-4x faster on a single decode pass, but its worker threads
            interact catastrophically with ``os.fork()``: the child inherits a
            CodecContext whose helper threads only exist in the parent, and
            ``avcodec_free_context`` then deadlocks in ``pthread_cond_wait``
            during normal cleanup of every per-call decode iterator. This
            shows up in any consumer that loads ``LazyVideo`` from a forked
            child (PyTorch ``DataLoader(num_workers>0)`` defaults to fork on
            Linux; Python ``multiprocessing`` and ``ProcessPoolExecutor``
            default to fork on Linux too). Single-threaded decode side-steps
            the issue entirely. If you are sure you will never decode from a
            forked child, set ``stream.thread_type = "AUTO"`` here for the
            speedup.
          * A single ``av.video.reformatter.VideoReformatter`` does
            colorspace + resize via libswscale; the per-frame ndarray is the
            already-formatted output (no extra colorspace copy).
        """
        av = _av_module()

        n_frames = len(frame_indices)
        if n_frames == 0:
            return self._empty_frames_array()

        n_channels = 3 if self.colorspace == "RGB" else 1

        # Resolve every requested frame to a (segment, local_index) tuple.
        seg_arr = np.empty(n_frames, dtype=np.int64)
        local_arr = np.empty(n_frames, dtype=np.int64)
        for i, fidx in enumerate(frame_indices):
            s, l = self._segment_for_frame(int(fidx))
            seg_arr[i] = s
            local_arr[i] = l

        # Sort by (segment, local_index) ascending. `order[k]` = original index
        # of the k-th element in sorted order; we'll scatter decoded frames
        # back to their original output position via this map.
        order = np.lexsort((local_arr, seg_arr))

        target_format = "rgb24" if self.colorspace == "RGB" else "gray8"

        frames: np.ndarray | None = None
        out_h: int | None = None
        out_w: int | None = None

        container = None
        stream = None
        decode_iter = None
        last_pts: int | None = None
        cur_seg = -1
        pts_table: np.ndarray | None = None
        reformatter = None
        # Same (segment, local_idx) can appear twice in one batch (duplicate
        # frame_indices). After the first decode the iterator has advanced past
        # that PTS; re-use the ndarray instead of calling next(decode_iter).
        seg_frame_cache: dict[int, np.ndarray] = {}

        try:
            for k, sorted_i in enumerate(order):
                sorted_i = int(sorted_i)
                seg_idx = int(seg_arr[sorted_i])
                local_idx = int(local_arr[sorted_i])

                if seg_idx != cur_seg:
                    if container is not None:
                        container.close()
                    container = av.open(self.video_files[seg_idx])
                    stream = container.streams.video[0]
                    # Single-threaded decode: avoids the libavcodec frame/slice
                    # thread + os.fork() deadlock during avcodec_free_context
                    # cleanup. See _load_frames docstring for details. Must be
                    # set before any decode call.
                    stream.thread_count = 1
                    stream.thread_type = "NONE"
                    pts_table = self._ensure_pts_table(seg_idx)
                    cur_seg = seg_idx
                    decode_iter = None
                    last_pts = None
                    seg_frame_cache = {}

                if local_idx in seg_frame_cache:
                    frames[sorted_i] = seg_frame_cache[local_idx]
                    continue

                target_pts = int(pts_table[local_idx])

                # Seek when:
                #   * we just opened the container,
                #   * the caller asked for a frame earlier than the decoder's
                #     current position (would otherwise overshoot),
                # otherwise we keep iterating the existing decode generator.
                need_seek = (
                    decode_iter is None
                    or last_pts is None
                    or target_pts < last_pts
                )
                if need_seek:
                    container.seek(
                        target_pts,
                        any_frame=False,
                        backward=True,
                        stream=stream,
                    )
                    decode_iter = container.decode(video=0)
                    last_pts = None
                    seg_frame_cache.clear()

                # Allocate the output array on the first frame of any segment
                # (first iteration overall, in practice).
                if frames is None:
                    if self.resize is not None:
                        out_h, out_w = self.resize
                    else:
                        out_h = stream.codec_context.height
                        out_w = stream.codec_context.width
                    if self.channel_format == "NCHW":
                        frames = np.zeros(
                            (n_frames, n_channels, out_h, out_w), dtype="uint8"
                        )
                    else:
                        frames = np.zeros(
                            (n_frames, out_h, out_w, n_channels), dtype="uint8"
                        )

                # Pull frames from the decoder until we land on target_pts.
                # `container.decode()` yields presentation-ordered frames, so a
                # PTS strictly greater than target means our target wasn't in
                # the stream (mismatched cache) -- accept the next available.
                collected = None
                while True:
                    try:
                        frame = next(decode_iter)
                    except StopIteration:
                        break
                    if frame.pts is None:
                        continue
                    last_pts = int(frame.pts)
                    if last_pts < target_pts:
                        continue
                    collected = frame
                    break

                if collected is None:
                    logging.warning(
                        "LazyVideo: end of segment %d reached early at "
                        "frame %d/%d (target pts=%d); leaving remaining "
                        "frames zero-filled.",
                        seg_idx,
                        k,
                        n_frames,
                        target_pts,
                    )
                    break

                if reformatter is None:
                    reformatter = av.video.reformatter.VideoReformatter()

                out_frame = reformatter.reformat(
                    collected,
                    width=out_w,
                    height=out_h,
                    format=target_format,
                )
                arr = out_frame.to_ndarray()
                if self.colorspace == "G":
                    # gray8 returns HxW; promote to HxWx1 to match channels.
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=-1)

                if self.channel_format == "NCHW":
                    arr = np.transpose(arr, (2, 0, 1))

                seg_frame_cache[local_idx] = arr
                frames[sorted_i] = arr
        finally:
            if container is not None:
                container.close()

        if frames is None:
            return self._empty_frames_array()
        return frames

    def to_hdf5(self, file: h5py.Group):
        r"""Save LazyVideo metadata and timestamps to an HDF5 group.

        The video file itself is not stored; only the path, timestamps,
        per-segment frame counts, optional per-segment PTS indices (so the
        decoder can do keyframe-aligned seeks without re-demuxing), and display
        options are saved. On load, the same video file path is used to open
        the video again.
        """
        file.attrs["object"] = self.__class__.__name__
        file.create_dataset("timestamps", data=self.timestamps)
        if len(self.video_files) == 1:
            file.attrs["video_file"] = str(self.video_files[0])
        else:
            dt = h5py.string_dtype(encoding="utf-8")
            file.create_dataset(
                "video_files", data=np.asarray(self.video_files, dtype=dt)
            )
        file.create_dataset(
            "segment_frame_counts",
            data=np.asarray(self.segment_frame_counts, dtype=np.int64),
        )
        # Persist any cached per-segment PTS arrays so a fresh process can
        # decode without re-walking packets. Stored as a vlen int64 dataset
        # plus a boolean "present" mask (so we can distinguish a genuinely
        # empty segment from a missing cache).
        if any(p is not None for p in self._pts_cache):
            vlen_dt = h5py.vlen_dtype(np.int64)
            ds = file.create_dataset(
                "segment_pts_indices",
                shape=(len(self._pts_cache),),
                dtype=vlen_dt,
            )
            present = np.zeros(len(self._pts_cache), dtype=bool)
            for i, p in enumerate(self._pts_cache):
                if p is None:
                    ds[i] = np.empty(0, dtype=np.int64)
                else:
                    ds[i] = np.asarray(p, dtype=np.int64)
                    present[i] = True
            file.create_dataset("segment_pts_present", data=present)
        file.attrs["colorspace"] = self.colorspace
        file.attrs["channel_format"] = self.channel_format
        if self.resize is None:
            file.attrs["resize"] = np.array([], dtype=np.int64)
        else:
            file.attrs["resize"] = np.array(self.resize, dtype=np.int64)

    @classmethod
    def from_hdf5(cls, file: h5py.Group) -> "LazyVideo":
        r"""Load a LazyVideo from an HDF5 group (metadata + timestamps only).

        The actual video is opened from the stored path when the object
        is used; ensure the path is valid on this machine.
        """
        if file.attrs.get("object") != cls.__name__:
            raise ValueError(
                f"File contains {file.attrs.get('object', 'unknown')}, "
                f"expected {cls.__name__}."
            )
        timestamps = file["timestamps"][:]
        if "video_files" in file:
            video_file = file["video_files"][:].astype(str).tolist()
        else:
            video_file = str(file.attrs["video_file"])
        colorspace = str(file.attrs["colorspace"])
        channel_format = str(file.attrs["channel_format"])
        r = file.attrs.get("resize")
        if r is None or (hasattr(r, "__len__") and len(r) == 0):
            resize = None
        else:
            resize = tuple(np.asarray(r).tolist())
        if "segment_frame_counts" in file:
            segment_frame_counts = file["segment_frame_counts"][:]
        else:
            segment_frame_counts = None
        if "segment_pts_indices" in file:
            raw = file["segment_pts_indices"][:]
            if "segment_pts_present" in file:
                present = file["segment_pts_present"][:]
            else:
                # Legacy: no present mask. Treat empty entries as missing.
                present = np.array([len(r_i) > 0 for r_i in raw], dtype=bool)
            segment_pts_indices: list[np.ndarray | None] | None = [
                np.asarray(raw[i], dtype=np.int64) if present[i] else None
                for i in range(len(raw))
            ]
        else:
            segment_pts_indices = None
        return cls(
            timestamps=timestamps,
            video_file=video_file,
            resize=resize,
            colorspace=colorspace,
            channel_format=channel_format,
            segment_frame_counts=segment_frame_counts,
            segment_pts_indices=segment_pts_indices,
        )
