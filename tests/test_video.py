import tempfile

import cv2
import numpy as np
import pytest

from temporaldata import LazyVideo


class TestLazyVideoLoader:

    num_frames = 100
    height = 64
    width = 48
    fps = 30

    @pytest.fixture
    def video_file(self):
        """Create a temporary video file that exists for the duration of the test"""

        # Create temp file (not directory)
        fd, video_path = tempfile.mkstemp(suffix=".mp4")
        import os

        os.close(fd)

        # Create video - NOTE: cv2.VideoWriter size is (width, height), not (height, width)!
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))

        for i in range(self.num_frames):
            frame = np.random.randint(
                0, 255, (self.height, self.width, 3), dtype=np.uint8
            )
            out.write(frame)

        out.release()

        yield video_path

        # Cleanup after test completes
        if os.path.exists(video_path):
            os.remove(video_path)

    @pytest.fixture
    def timestamps(self):
        return np.arange(self.num_frames) / 1.0  # cast to float

    @pytest.mark.parametrize("start,end", [(0, 10), (10, 25), (31, 37)])
    def test_basic(self, video_file, timestamps, start, end):
        """Test slicing at different positions"""
        video = LazyVideo(
            timestamps=timestamps,
            video_file=video_file,
            resize=None,
            colorspace="RGB",
            channel_format="NCHW",
        )
        samples = video.slice(start, end)
        assert samples.frames.shape == (end - start, 3, self.height, self.width)

    @pytest.mark.parametrize("channel_format", ["NCHW", "NHWC"])
    def test_grayscale(self, video_file, timestamps, channel_format):
        """Test grayscale for different channel formats"""

        video = LazyVideo(
            timestamps=timestamps,
            video_file=video_file,
            resize=None,
            colorspace="G",
            channel_format=channel_format,
        )
        num_frames = 10
        samples = video.slice(0, num_frames)
        if channel_format == "NCHW":
            assert samples.frames.shape == (num_frames, 1, self.height, self.width)
        else:
            assert samples.frames.shape == (num_frames, self.height, self.width, 1)

    @pytest.mark.parametrize("colorspace", ["G", "RGB"])
    @pytest.mark.parametrize("channel_format", ["NCHW", "NHWC"])
    def test_resize(self, video_file, timestamps, channel_format, colorspace):
        """Test resizing with different colorspaces and channel formats"""

        resize_ = 64
        num_frames = 10
        video = LazyVideo(
            timestamps=timestamps,
            video_file=video_file,
            resize=(resize_, resize_),
            colorspace=colorspace,
            channel_format=channel_format,
        )
        samples = video.slice(0, num_frames)
        if channel_format == "NCHW":
            if colorspace == "RGB":
                assert samples.frames.shape == (num_frames, 3, resize_, resize_)
            else:
                assert samples.frames.shape == (num_frames, 1, resize_, resize_)
        else:
            if colorspace == "RGB":
                assert samples.frames.shape == (num_frames, resize_, resize_, 3)
            else:
                assert samples.frames.shape == (num_frames, resize_, resize_, 1)

    def test_video_exists(self, video_file, timestamps):
        """Test correct error is raised if video does not exist"""
        video_file_bad = video_file + "_bad.mp4"
        with pytest.raises(IOError):
            LazyVideo(timestamps=timestamps, video_file=video_file_bad)

    def test_timestamp_check(self, video_file, timestamps):
        """Test correct error is raised if timestamps don't match frames"""
        timestamps_bad = np.arange(timestamps.shape[0] + 10) / 1.0
        with pytest.raises(RuntimeError):
            LazyVideo(timestamps=timestamps_bad, video_file=video_file)

    def test_option_validation(self, video_file, timestamps):
        """Test correct error is raised if bad constructor args are passed"""

        # bad resize args
        with pytest.raises(ValueError):
            LazyVideo(timestamps=timestamps, video_file=video_file, resize=64)

        with pytest.raises(ValueError):
            LazyVideo(timestamps=timestamps, video_file=video_file, resize=(23, 23, 23))

        # bad colorspace args
        with pytest.raises(ValueError):
            LazyVideo(timestamps=timestamps, video_file=video_file, colorspace="BGR")

        # bad channel_format args
        with pytest.raises(ValueError):
            LazyVideo(
                timestamps=timestamps, video_file=video_file, channel_format="NHCW"
            )
