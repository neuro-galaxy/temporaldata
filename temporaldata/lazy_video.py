from __future__ import annotations

import subprocess
from typing import Sequence
import logging

import h5py
import numpy as np

from .irregular_ts import IrregularTimeSeries

_cv2 = None


def _cv2_module():
    """Lazy-import OpenCV once; do not store the module on LazyVideo instances (breaks copy.deepcopy)."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as _cv2_mod

            _cv2 = _cv2_mod
        except ImportError:
            raise ImportError(
                "OpenCV not installed, you must install temporaldata using "
                "`pip install -e .[video]`"
            )
    return _cv2


def _probe_segment_frame_count(path: str) -> int:
    """Run `ffprobe -count_packets` to get an exact video frame count.

    This is expensive (full demux pass) and should only be called when a
    cached count is not available.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-count_packets", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


class LazyVideo(object):
    r"""An object that lazily loads batches of video data using OpenCV.

    Args:
        timestamps: array of camera timestamps
        video_file: absolute path to video file
        resize: (height, width) to resize the frames to (or None to keep original size)
        colorspace: "RGB" | "G"
        channel_format: "NCHW" | "NHWC"

    """

    def __init__(
        self,
        timestamps: np.ndarray,
        video_file: str | Sequence[str],
        resize: tuple | None = None,
        colorspace: str = "RGB",
        channel_format: str = "NCHW",
        segment_frame_counts: np.ndarray | Sequence[int] | None = None,
    ):
        # Validate cv2 is importable, but do not actually open captures here.
        # Captures are opened lazily per-slice in `_load_frames` so that:
        #   * objects are picklable (multi-worker DataLoader-friendly),
        #   * file descriptors don't pile up across many sessions,
        #   * a long-running job self-heals from transient I/O / decoder
        #     state corruption (e.g. matroska "Read error at pos." cascades).
        _cv2_module()

        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        if isinstance(video_file, str):
            video_files = [video_file]
        else:
            video_files = [str(path) for path in video_file]
        if len(video_files) == 0:
            raise ValueError("At least one video file must be provided.")
        self.video_files = video_files
        # Backward-compatible public attribute used in older code paths.
        self.video_file = video_files[0] if len(video_files) == 1 else list(video_files)

        if (resize is None) or (isinstance(resize, tuple) and len(resize) == 2):
            self.resize = resize
        else:
            raise ValueError('"resize" arg must be None or a tuple (height, width)')

        if colorspace == "RGB" or colorspace == "G":
            self.colorspace = colorspace
        else:
            raise ValueError('"colorspace" arg must be "RGB" or "G"')

        if channel_format == "NCHW" or channel_format == "NHWC":
            self.channel_format = channel_format
        else:
            raise ValueError('"channel_format" arg must be "NCHW" or "NHWC"')

        if segment_frame_counts is None:
            # Slow path: probe each segment for an exact frame count. This walks
            # the entire packet stream and can take seconds-to-minutes per file.
            # Prefer caching counts at ingest (`to_hdf5` writes them) so this
            # branch is only taken for legacy data or fresh in-memory construction.
            print(f"[WARNING] cache miss for segment frame counts for {self.video_files}")
            segment_frame_counts = [
                _probe_segment_frame_count(path) for path in self.video_files
            ]
        else:
            segment_frame_counts = list(segment_frame_counts)
            if len(segment_frame_counts) != len(self.video_files):
                raise ValueError(
                    f"segment_frame_counts has length {len(segment_frame_counts)} "
                    f"but there are {len(self.video_files)} video files."
                )

        self.segment_frame_counts = np.asarray(segment_frame_counts, dtype=np.int64)
        self.segment_frame_offsets = np.cumsum(
            np.concatenate(([0], self.segment_frame_counts[:-1]))
        )
        frame_count = int(self.segment_frame_counts.sum())
        if frame_count != self.timestamps.shape[0]:
            if frame_count > self.timestamps.shape[0]:
                frame_count = self.timestamps.shape[0] # If we can just drop the extra frames, we should do that.
            else:
                raise ValueError(f"Frame count mismatch: {frame_count} != {self.timestamps.shape[0]}") # TODO: check this
        self.frame_count = frame_count

        self.frame_indices = np.arange(frame_count, dtype=np.int64)

        if frame_count > 1 and np.any(np.diff(self.timestamps[:frame_count]) < 0):
            sort_idx = np.argsort(self.timestamps[:frame_count])
            self.timestamps[:frame_count] = self.timestamps[sort_idx]
            self.frame_indices = sort_idx.astype(np.int64)
            logging.info(
                "LazyVideo: sorted %d timestamps that were not monotonically increasing",
                frame_count,
            )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        vf = self.video_files[0] if len(self.video_files) == 1 else list(self.video_files)
        dup = self.__class__(
            timestamps=self.timestamps.copy(),
            video_file=vf,
            resize=self.resize,
            colorspace=self.colorspace,
            channel_format=self.channel_format,
            segment_frame_counts=self.segment_frame_counts.copy(),
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
        return cls(
            timestamps=timestamps,
            video_file=video_files,
            resize=first.resize,
            colorspace=first.colorspace,
            channel_format=first.channel_format,
        )

    def _segment_for_frame(self, frame_index: int):
        segment_idx = int(
            np.searchsorted(self.segment_frame_offsets, frame_index, side="right") - 1
        )
        segment_start = int(self.segment_frame_offsets[segment_idx])
        local_index = int(frame_index - segment_start)
        return segment_idx, local_index

    def slice(self, start: float, end: float):
        r"""Returns a new :obj:`IrregularTimeSeries` object that contains the data
        between the start and end times. The end time is exclusive, the slice will
        only include data in :math:`[\textrm{start}, \textrm{end})`.

        Args:
            start: Start time.
            end: End time.

        """

        timestamps = IrregularTimeSeries(
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            frame_indices=self.frame_indices,
            domain="auto",
        )
        timestamps_sliced = timestamps.slice(start=start, end=end)
        frames_sliced = self._load_frames(timestamps_sliced.frame_indices)

        timestamps_sliced.frames = frames_sliced

        return timestamps_sliced

    def _load_frames(self, frame_indices: np.ndarray):
        cv2 = _cv2_module()

        n_frames = len(frame_indices)
        n_channels = 3 if self.colorspace == "RGB" else 1

        if n_frames == 0:
            if self.channel_format == "NCHW":
                return np.zeros((0, n_channels, 1, 1), dtype="uint8")
            return np.zeros((0, 1, 1, n_channels), dtype="uint8")

        is_contiguous = bool(
            np.sum(np.diff(frame_indices)) == (len(frame_indices) - 1)
        )

        # Open each touched segment once for the duration of this call, and
        # release everything at the end (success or failure). Captures are not
        # cached across calls -- this gives us a fresh decoder state each
        # slice, which avoids long-running ffmpeg state corruption (e.g. the
        # matroska "Read error at pos." cascade) and keeps the object
        # picklable for multi-worker DataLoaders.
        open_captures: dict[int, "cv2.VideoCapture"] = {}

        def _get_capture(segment_idx: int) -> "cv2.VideoCapture":
            cap = open_captures.get(segment_idx)
            if cap is None:
                path = self.video_files[segment_idx]
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise IOError(f"Error opening video file {path}")
                open_captures[segment_idx] = cap
            return cap

        frames = None
        previous_segment_idx = None
        previous_local_idx = None

        try:
            for fr, frame_index in enumerate(frame_indices):
                segment_idx, local_index = self._segment_for_frame(int(frame_index))
                video_capture = _get_capture(segment_idx)
                should_seek = (
                    fr == 0
                    or not is_contiguous
                    or previous_segment_idx != segment_idx
                    or previous_local_idx is None
                    or local_index != previous_local_idx + 1
                )
                if should_seek:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, local_index)
                ret, frame = video_capture.read()
                if not ret:
                    if fr == 0:
                        # Caller is responsible for handling this; we do NOT
                        # silently return a 0-length array because that
                        # produces a confusing shape mismatch downstream
                        # (frames first-dim 0 vs timestamps first-dim N).
                        raise RuntimeError(
                            f"VideoCapture.read() returned False on first frame "
                            f"of slice (segment={segment_idx} "
                            f"path={self.video_files[segment_idx]} "
                            f"local_index={local_index})."
                        )
                    logging.warning(
                        "LazyVideo: reached end of video early at frame %d/%d; "
                        "returning blank frames for the remainder.",
                        fr,
                        n_frames,
                    )
                    break

                if fr == 0:
                    if self.resize is None:
                        height, width, _ = frame.shape
                    else:
                        height, width = self.resize

                    if self.channel_format == "NCHW":
                        frames = np.zeros(
                            (n_frames, n_channels, height, width), dtype="uint8"
                        )
                    else:  # NHWC
                        frames = np.zeros(
                            (n_frames, height, width, n_channels), dtype="uint8"
                        )

                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)

                if self.colorspace == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:  # "G"
                    frame = np.expand_dims(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=-1
                    )

                if self.channel_format == "NCHW":
                    frame = np.transpose(frame, (2, 0, 1))

                frames[fr] = frame
                previous_segment_idx = segment_idx
                previous_local_idx = local_index
        finally:
            for cap in open_captures.values():
                cap.release()

        if frames is None:
            if self.channel_format == "NCHW":
                frames = np.zeros((0, n_channels, 1, 1), dtype="uint8")
            else:
                frames = np.zeros((0, 1, 1, n_channels), dtype="uint8")
        return frames

    def to_hdf5(self, file: h5py.Group):
        r"""Save LazyVideo metadata and timestamps to an HDF5 group.

        The video file itself is not stored; only the path, timestamps,
        per-segment frame counts (so loaders don't have to re-probe with
        ffprobe), and display options are saved. On load, the same video
        file path is used to open the video again (path may be relative
        or absolute).
        """
        file.attrs["object"] = self.__class__.__name__
        file.create_dataset("timestamps", data=self.timestamps)
        if len(self.video_files) == 1:
            file.attrs["video_file"] = str(self.video_files[0])
        else:
            dt = h5py.string_dtype(encoding="utf-8")
            file.create_dataset("video_files", data=np.asarray(self.video_files, dtype=dt))
        # Cache per-segment frame counts. Avoids the multi-second
        # `ffprobe -count_packets` pass on every load / every worker.
        file.create_dataset(
            "segment_frame_counts",
            data=np.asarray(self.segment_frame_counts, dtype=np.int64),
        )
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
        # Legacy files written before segment_frame_counts caching was added
        # will fall back to running ffprobe in __init__.
        if "segment_frame_counts" in file:
            segment_frame_counts = file["segment_frame_counts"][:]
        else:
            segment_frame_counts = None
        return cls(
            timestamps=timestamps,
            video_file=video_file,
            resize=resize,
            colorspace=colorspace,
            channel_format=channel_format,
            segment_frame_counts=segment_frame_counts,
        )

