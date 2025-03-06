import time
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


class VideoLoop:
    cap: cv2.VideoCapture
    fps: int
    frame_count: int
    width: int
    height: int
    frame_time: float
    video_resolution: tuple[int, int]
    last_frame_time: float

    def __init__(
        self,
        video_path: str | Path,
        loop: bool = False,
        skip_seconds: float = 0.0,
    ) -> None:
        self.video_path = video_path
        self.loop = loop
        self.skip_seconds = skip_seconds

    def __iter__(self) -> "VideoLoop":
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        """
        Read the next frame from the video and calculate the time to wait

        Returns
            tuple[int, np.ndarray]:
                - time to wait in milliseconds
                - frame read from the video
        """
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.__next__()
            else:
                self.cap.release()
                raise StopIteration

        # calculate the time elapsed since the last frame
        current_frame_time = time.time()
        dt = (current_frame_time - self.last_frame_time) * 1000
        self.last_frame_time = current_frame_time

        sleep_time = max(1, int(self.frame_time - dt))

        return sleep_time, frame

    def __del__(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoLoop":
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Error: cannot read video file {self.video_path}")

        # get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # calculate utility variables
        self.frame_time = 1000 / self.fps
        self.last_frame_time = time.time()
        self.video_resolution = (self.width, self.height)

        if self.skip_seconds > self.frame_count / self.fps:
            raise ValueError(
                f"Error: skip_seconds ({self.skip_seconds:.2f}s) is greater than the video duration ({self.frame_count / self.fps:.2f}s)"
            )

        self.reset()

        return self

    def __exit__(
        self,
        _: type[BaseException] | None,
        __: BaseException | None,
        ___: TracebackType | None,
    ) -> None:
        self.cap.release()

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.fps * self.skip_seconds))
        self.last_frame_time = time.time()
