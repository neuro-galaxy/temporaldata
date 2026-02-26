from __future__ import annotations

from typing import Sequence
import logging

import h5py
import numpy as np

from .irregular_ts import IrregularTimeSeries


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
    ):

        try:
            import cv2

            self.cv2 = cv2  # Store for use in other methods
        except ImportError:
            raise ImportError(
                "OpenCV not installed, you must install temporaldata using "
                "`pip install -e .[video]`"
            )

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

        self.video_captures = []
        segment_frame_counts = []
        for path in self.video_files:
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                for opened_capture in self.video_captures:
                    opened_capture.release()
                raise IOError(f"Error opening video file {path}")
            self.video_captures.append(capture)
            segment_frame_counts.append(
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            )

        self.segment_frame_counts = np.asarray(segment_frame_counts, dtype=np.int64)
        self.segment_frame_offsets = np.cumsum(
            np.concatenate(([0], self.segment_frame_counts[:-1]))
        )
        frame_count = int(self.segment_frame_counts.sum())
        if frame_count != self.timestamps.shape[0]:
            raise RuntimeError(
                f"Video frames ({frame_count}) do not match timestamps ({self.timestamps.shape[0]})"
            )
        self.frame_count = frame_count

        self.frame_indices = np.arange(frame_count, dtype=np.int64)

    def __del__(self):
        r"""Close video capture upon object destruction"""
        for capture in getattr(self, "video_captures", []):
            capture.release()

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

        is_contiguous = np.sum(np.diff(frame_indices)) == (len(frame_indices) - 1)
        n_frames = len(frame_indices)
        n_channels = 3 if self.colorspace == "RGB" else 1
        frames = None
        previous_segment_idx = None
        previous_local_idx = None

        for fr, frame_index in enumerate(frame_indices):
            segment_idx, local_index = self._segment_for_frame(int(frame_index))
            video_capture = self.video_captures[segment_idx]
            should_seek = (
                fr == 0
                or not is_contiguous
                or previous_segment_idx != segment_idx
                or previous_local_idx is None
                or local_index != previous_local_idx + 1
            )
            if should_seek:
                video_capture.set(self.cv2.CAP_PROP_POS_FRAMES, local_index)
            ret, frame = video_capture.read()
            if ret:
                # create frames array for first frame
                if fr == 0:
                    if self.resize is None:
                        height, width, _ = frame.shape
                    else:
                        height, width = self.resize

                    if self.channel_format == "NCHW":
                        frames = np.zeros(
                            (n_frames, n_channels, height, width), dtype="uint8"
                        )
                    elif self.channel_format == "NHWC":
                        frames = np.zeros(
                            (n_frames, height, width, n_channels), dtype="uint8"
                        )

                # modify frame data
                if self.resize is not None:
                    frame = self.cv2.resize(frame, self.resize)

                if self.colorspace == "RGB":
                    frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                elif self.colorspace == "G":
                    # keep color channel
                    frame = np.expand_dims(
                        self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY), axis=-1
                    )

                if self.channel_format == "NCHW":
                    frame = np.transpose(frame, (2, 0, 1))

                # save frame data into array
                frames[fr] = frame
                previous_segment_idx = segment_idx
                previous_local_idx = local_index

            else:
                if fr == 0:
                    # First read failed (e.g. seek error); return empty so caller gets no frames
                    if self.channel_format == "NCHW":
                        frames = np.zeros((0, n_channels, 1, 1), dtype="uint8")
                    else:
                        frames = np.zeros((0, 1, 1, n_channels), dtype="uint8")
                else:
                    print(
                        "warning! reached end of video; "
                        + "returning blank frames for remainder of requested indices"
                    )
                break

        if frames is None:
            if self.channel_format == "NCHW":
                frames = np.zeros((0, n_channels, 1, 1), dtype="uint8")
            else:
                frames = np.zeros((0, 1, 1, n_channels), dtype="uint8")
        return frames

    def to_hdf5(self, file: h5py.Group):
        r"""Save LazyVideo metadata and timestamps to an HDF5 group.

        The video file itself is not stored; only the path, timestamps, and
        display options are saved. On load, the same video file path is used
        to open the video again (path may be relative or absolute).
        """
        file.attrs["object"] = self.__class__.__name__
        file.create_dataset("timestamps", data=self.timestamps)
        if len(self.video_files) == 1:
            file.attrs["video_file"] = str(self.video_files[0])
        else:
            dt = h5py.string_dtype(encoding="utf-8")
            file.create_dataset("video_files", data=np.asarray(self.video_files, dtype=dt))
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
        return cls(
            timestamps=timestamps,
            video_file=video_file,
            resize=resize,
            colorspace=colorspace,
            channel_format=channel_format,
        )

