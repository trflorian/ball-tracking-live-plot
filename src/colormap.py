import numpy as np
import cv2

def colormap_rainbow(value: float) -> tuple[int, int, int]:
    """Map a value between 0 and 1 to a color in the rainbow colormap."""
    pixel_img = np.array([[value * 255]], dtype=np.uint8) # shape: (1, 1)
    pixel_cmap_img = cv2.applyColorMap(pixel_img, cv2.COLORMAP_RAINBOW) # shape: (1, 1, 3)
    return pixel_cmap_img.flatten().tolist()

def visualize_colormap():
    target_shape = (500, 50, 3)

    value_range = np.linspace(0, 1, target_shape[0])

    colormap_strip_1d = np.array([colormap_rainbow(value) for value in value_range])
    colormap_strip_2d = np.expand_dims(colormap_strip_1d, axis=0).repeat(target_shape[1], axis=0).astype(np.uint8)

    cv2.imshow("Rainbow Colormap", colormap_strip_2d)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_colormap()