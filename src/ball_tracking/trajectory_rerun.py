import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ball_tracking.core import Point2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("media/ball.mp4"),
        help="Path to the video file",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()

    rr.init("ball_tracking", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimeSeriesView(
                origin="/ball/position_y", axis_y=rrb.ScalarAxis(range=[0, 700])
            ),
            rrb.TimeSeriesView(
                origin="/ball/velocity_y", axis_y=rrb.ScalarAxis(range=[-200, 200])
            ),
            rrb.TimeSeriesView(
                origin="/ball/acceleration_y", axis_y=rrb.ScalarAxis(range=[-30, 10])
            ),
            rrb.Spatial2DView(origin="/ball/trajectory"),
        )
    )

    cap = cv2.VideoCapture(str(args.video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    max_time = 1.2
    num_frames = int(max_time * fps)

    skip_time = 0.2
    skip_frames = int(skip_time * fps)

    # initialize background model
    bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    ret, frame0 = cap.read()
    if not ret:
        logger.error("Error: cannot read video file")
        exit(1)
    bg_sub.apply(frame0, learningRate=1.0)

    tracked_pos: list[Point2D] = []

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_annotated = frame.copy()

        frame_index += 1
        if frame_index >= num_frames:
            break

        rr.set_time("frame_idx", sequence=frame_index)

        # filter based on color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(
            hsv,
            lowerb=np.array([20, 0, 0]),
            upperb=np.array([100, 255, 255]),
        )

        # filter based on motion
        mask_fg = bg_sub.apply(frame, learningRate=0)

        # combine both masks
        mask = cv2.bitwise_and(mask_color, mask_fg)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        )

        # find largest contour corresponding to the ball we want to track
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w // 2, y + h // 2)
            tracked_pos.append(center)

            cv2.circle(frame_annotated, center, 30, (255, 0, 0), 2)
            cv2.circle(frame_annotated, center, 2, (255, 0, 0), 2)

        if len(tracked_pos) > 0:
            pos = np.array([tracked_pos[0][1] - pos[1] for pos in tracked_pos])
            vel = np.diff(pos)
            acc = np.diff(vel)

            rr.log("ball/position_y", rr.Scalars(float(pos[-1])))

            if vel.size > 0:
                dv = vel[-1]
                rr.log("ball/velocity_y", rr.Scalars(float(dv)))
            if acc.size > 0:
                da = acc[-1]
                rr.log("ball/acceleration_y", rr.Scalars(float(da)))

        # draw trajectory
        for i in range(1, len(tracked_pos)):
            cv2.line(
                frame_annotated, tracked_pos[i - 1], tracked_pos[i], (255, 0, 0), 1
            )

        # cv2.imshow("Frame", frame_annotated)
        frame_annotated_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
        rr.log("frame", rr.Image(frame_annotated_rgb))

    cap.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
