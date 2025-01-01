import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self):
        self.pos = None
        self.vel = None
        self.acc = None

    def update(self, pos, vel, acc, dt):
        if self.pos is None:
            self.pos = pos
        if self.vel is None:
            self.vel = vel
        if self.acc is None:
            self.acc = acc

        pred_next = self.predict_states(dt, 1)

        a = 0.5
        self.pos = a * pred_next[0, 0] + (1 - a) * pos
        self.vel = a * pred_next[0, 1] + (1 - a) * vel
        self.acc = a * pred_next[0, 2] + (1 - a) * acc

    def get_transition_matrix(self, dt):
        return np.array(
            [
                [1, dt, 0],
                [0, 1, dt],
                [0, 0, 1],
            ]
        )

    def get_state_vec(self):
        return np.array(
            [
                self.pos,
                self.vel,
                self.acc,
            ]
        )

    def predict_states(self, dt_step, steps):
        predictions = []
        curr_state = self.get_state_vec()
        A = self.get_transition_matrix(dt_step)
        for _ in range(steps):
            curr_state = np.matmul(A, curr_state)
            predictions.append(curr_state)
        return np.array(predictions)


def main():
    cap = cv2.VideoCapture("media/ball.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # initialize background model
    bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)
    ret, frame0 = cap.read()
    if not ret:
        print("Error: cannot read video file")
        exit(1)
    bg_sub.apply(frame0, learningRate=1.0)

    tracked_pos = []

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 2), dpi=100)

    axs[0].set_title("Position")
    axs[0].set_ylim(0, 700)
    axs[1].set_title("Velocity")
    axs[1].set_ylim(-200, 200)
    axs[2].set_title("Acceleration")
    axs[2].set_ylim(-30, 10)

    pl_pos = axs[0].plot([], [], c="b")[0]
    pl_vel = axs[1].plot([], [], c="b")[0]
    pl_acc = axs[2].plot([], [], c="b")[0]

    pl_pos_pred = axs[0].plot([], [], c="g", linestyle="--")[0]
    pl_vel_pred = axs[1].plot([], [], c="g", linestyle="--")[0]
    pl_acc_pred = axs[2].plot([], [], c="g", linestyle="--")[0]

    for ax in axs:
        ax.set_xlim(0, 20)
        ax.grid(True)

    fig.canvas.draw()
    bg_axs = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axs]

    kf = KalmanFilter()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracked_pos.clear()
            continue
        frame_annotated = frame.copy()

        st = time.time()

        # filter based on color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, (20, 0, 0), (100, 255, 255))

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

        # draw trajectory
        for i in range(1, len(tracked_pos)):
            cv2.line(
                frame_annotated, tracked_pos[i - 1], tracked_pos[i], (255, 0, 0), 1
            )

        row1 = cv2.hconcat([frame, cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)])
        row2 = cv2.hconcat(
            [
                cv2.cvtColor(mask_fg, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            ]
        )
        composite = cv2.vconcat([row1, row2])
        composite = cv2.resize(composite, (0, 0), fx=0.75, fy=0.75)

        pos = np.array([tracked_pos[0][1] - pos[1] for pos in tracked_pos])
        vel = np.diff(pos)
        acc = np.diff(vel)

        if len(pos) > 0 and len(vel) > 0 and len(acc) > 0:
            if len(contours) > 0:
                kf.update(pos[-1], vel[-1], acc[-1], dt=1)

            steps = 20
            preds_future = kf.predict_states(dt_step=1, steps=steps)
            preds_past = kf.predict_states(dt_step=-1, steps=steps)[::-1]
            preds = np.concat([preds_past, [kf.get_state_vec()], preds_future])

            t0 = len(pos)
            t = t0 + np.array(range(steps * 2 + 1)) - steps - 1
            pl_pos_pred.set_data(t, preds[:, 0])
            pl_vel_pred.set_data(t, preds[:, 1])
            pl_acc_pred.set_data(t, preds[:, 2])

        pl_pos.set_data(range(len(pos)), pos)
        pl_vel.set_data(range(len(vel)), vel)
        pl_acc.set_data(range(len(acc)), acc)

        fig.canvas.restore_region(bg_axs[0])
        axs[0].draw_artist(pl_pos)
        axs[0].draw_artist(pl_pos_pred)
        fig.canvas.blit(axs[0].bbox)

        fig.canvas.restore_region(bg_axs[1])
        axs[1].draw_artist(pl_vel)
        axs[1].draw_artist(pl_vel_pred)
        fig.canvas.blit(axs[1].bbox)

        fig.canvas.restore_region(bg_axs[2])
        axs[2].draw_artist(pl_acc)
        axs[2].draw_artist(pl_acc_pred)
        fig.canvas.blit(axs[2].bbox)

        buf = fig.canvas.buffer_rgba()
        plot = np.asarray(buf)
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

        # pad plot with white to match width of frame, same left and right padding
        pad = (frame.shape[1] - plot.shape[1]) // 2
        plot = cv2.copyMakeBorder(
            plot, 50, 50, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        # vstack frame and plot
        frame_annotated = cv2.vconcat([plot, frame_annotated])

        cv2.imshow("Frame", frame_annotated)
        # cv2.imshow("Plot", plot)

        et = time.time()

        dt = (et - st) * 1000
        target_time = 1000 / fps

        sleep_time = max(1, int(target_time - dt))

        key = cv2.waitKey(sleep_time) & 0xFF
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
