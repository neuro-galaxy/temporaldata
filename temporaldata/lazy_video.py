from __future__ import annotations

from typing import Dict, List, Union
import logging

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
        video_file: str,
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

        self.timestamps = timestamps
        self.video_file = video_file

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

        self.video_capture = cv2.VideoCapture(video_file)

        if not self.video_capture.isOpened():
            raise IOError(f"Error opening video file {video_file}")

        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count != self.timestamps.shape[0]:
            raise RuntimeError(
                f"Video frames ({frame_count}) do not match timestamps ({self.timestamps.shape[0]})"
            )
        self.frame_count = frame_count

        self.frame_indices = np.arange(frame_count)

    def __del__(self):
        r"""Close video capture upon object destruction"""
        self.video_capture.release()

    def __len__(self):
        r"""Returns the first dimension of timestamps."""
        return self.frame_count

    def __repr__(self):
        cls = self.__class__.__name__
        info = ",\n".join(
            [f"timestamps=[{self.frame_count}]", f"frames=[{self.frame_count}]"]
        )
        return f"{cls}(\n{info}\n)"

    def slice(self, start: float, end: float):
        r"""Returns a new :obj:`IrregularTimeSeries` object that contains the data
        between the start and end times. The end time is exclusive, the slice will
        only include data in :math:`[\textrm{start}, \textrm{end})`.

        Args:
            start: Start time.
            end: End time.

        """

        timestamps = IrregularTimeSeries(
            timestamps=self.timestamps,
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

        for fr, i in enumerate(frame_indices):
            if fr == 0 or not is_contiguous:
                self.video_capture.set(1, i)
            ret, frame = self.video_capture.read()
            if ret:
                # create frames array for first frame
                if fr == 0:
                    if self.resize is None:
                        height, width, _ = frame.shape
                    else:
                        height, width = self.resize

                    if self.colorspace == "RGB":
                        n_channels = 3
                    elif self.colorspace == "G":
                        n_channels = 1

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

            else:
                print(
                    "warning! reached end of video; "
                    + "returning blank frames for remainder of requested indices"
                )
                break

        return frames

