from collections import deque
from itertools import pairwise

from pathlib import Path

import time
import cv2

from colormap import colormap_rainbow


def main():
    use_alpha_blending = False
    trajectory_length = 40
    skip_seconds = 4.5
    looping = False

    video_path = Path("media/ball3.mp4")

    cap = cv2.VideoCapture(str(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_resolution = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    video_writer = cv2.VideoWriter(
        video_path.with_name(video_path.stem + "_tracked.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        video_resolution,
    )

    # initialize background model
    bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=128, detectShadows=False)

    tracked_pos = deque(maxlen=trajectory_length)

    def reset():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * skip_seconds))
        tracked_pos.clear()

    reset()
    ret, frame0 = cap.read()
    if not ret:
        print("Error: cannot read video file")
        exit(1)
    bg_sub.apply(frame0, learningRate=1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            if looping:
                reset()
                continue
            else:
                break
        frame_annotated = frame.copy()

        st = time.time()

        # filter based on color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, (30, 30, 30), (100, 150, 150))
        mask_color = cv2.morphologyEx(
            mask_color,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )

        # filter based on motion
        mask_fg = bg_sub.apply(frame, learningRate=0)
        mask_fg = cv2.dilate(mask_fg, None, iterations=2)

        # combine both masks
        mask = cv2.bitwise_and(mask_color, mask_fg)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        # find largest contour corresponding to the ball we want to track
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w // 2, y + h // 2)

            if len(tracked_pos) > 0:
                # smooth the trajectory
                prev_center = tracked_pos[-1]
                alpha = 0.9
                center = (
                    int((1 - alpha) * prev_center[0] + alpha * center[0]),
                    int((1 - alpha) * prev_center[1] + alpha * center[1]),
                )

            tracked_pos.append(center)

            cv2.circle(frame_annotated, center, 30, (255, 0, 0), 2)
            cv2.circle(frame_annotated, center, 2, (255, 0, 0), 2)

        # draw trajectory
        traj_len = len(tracked_pos)
        for i, (p1, p2) in enumerate(pairwise(tracked_pos)):
            norm_idx = i / traj_len
            color = colormap_rainbow(norm_idx)

            if use_alpha_blending:
                # Create a temporary image to draw the line
                temp = frame_annotated.copy()
                cv2.line(temp, pt1=p1, pt2=p2, color=color, thickness=2)

                # Blend the temporary image with the original frame
                cv2.addWeighted(
                    temp, norm_idx, frame_annotated, 1 - norm_idx, 0, frame_annotated
                )
            else:
                cv2.line(frame_annotated, pt1=p1, pt2=p2, color=color, thickness=2)
        
        video_writer.write(frame_annotated)

        cv2.imshow("Frame", frame_annotated)

        cv2.imshow("Mask FG", mask_fg)
        cv2.imshow("Mask Color", mask_color)

        et = time.time()

        dt = (et - st) * 1000
        target_time = 1000 / fps

        sleep_time = max(1, int(target_time - dt))

        key = cv2.waitKey(sleep_time) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            reset()

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
