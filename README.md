# Ball Tracking with Live Trajectory Visualization

In this project I showcase how you can create an animated plot wiht OpenCV and Matplotlib. To demonstrate the real-time animated plotting, I am tracking a ball that is thrown vertically into the air. The ball's vertical position, velocity and acceleration are plotted in the figure.

![2024-11-07_23-31-58-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/128367fc-f6f3-4c45-8405-e146bd22148c)

## 🌟 Quickstart

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for setup. Simply sync the project dependencies and then run the `tracker.py` file to get started!

```Shell
uv sync
uv run tracker.py
```

## 📈 Matplotlib + OpenCV

To get a plot/figure from matplotlib into OpenCV, I render the canvas into a buffer in memory, store the buffer in a numpy array and transform it to the correct BGR format to display in OpenCV.

```Python
fig.canvas.draw()

buf = fig.canvas.buffer_rgba()
plot = np.asarray(buf)
plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
```

## 🎨 Blitting
The naive draw call in matplotlib as shown above is quite expensive, the full figure needs to be re-drawn every frame. To improve the performance, I make use of a technique called [blitting](https://matplotlib.org/stable/users/explain/animations/blitting.html). 
This allos me to only re-draw regions that have changed and therefore drastically redducing the rendering time.

```Python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 2), dpi=100)

...

# initialize empty plots
pl_pos = axs[0].plot([], [], c="b")[0]
pl_vel = axs[1].plot([], [], c="b")[0]
pl_acc = axs[2].plot([], [], c="b")[0]

# draw initial backgrounds
fig.canvas.draw()
bg_axs = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axs]

while True:
  ...

  # Update plots
  pl_pos.set_data(range(len(pos)), pos)
  pl_vel.set_data(range(len(vel)), vel)
  pl_acc.set_data(range(len(acc)), acc)
  
  # Blit Pos
  fig.canvas.restore_region(bg_axs[0])
  axs[0].draw_artist(pl_pos)
  fig.canvas.blit(axs[0].bbox)
  
  # Blit Vel
  fig.canvas.restore_region(bg_axs[1])
  axs[1].draw_artist(pl_vel)
  fig.canvas.blit(axs[1].bbox)
  
  # Blit Acc
  fig.canvas.restore_region(bg_axs[2])
  axs[2].draw_artist(pl_acc)
  fig.canvas.blit(axs[2].bbox)

  # show plot, cv2.waitKey etc.
  ...
```

## 🎭 Visualization of the Masks
![Screencastfrom11-07-2024103538PM-ezgif com-cut](https://github.com/user-attachments/assets/9209500a-94f4-4670-be64-c332dc839801)
