import numpy as np # python.exe -m pip install numpy
from scipy import interpolate # python.exe -m pip install scipy
import cv2 # python.exe -m pip install opencv-python

image_path = r"tiger.jpg"
method = "Bilinear" # Nearest, Bilinear, Bicubic
scale = 2

# Read image
image = cv2.imread(image_path) 
dimension = image.shape[:-1]
cv2.imshow("Original", image)

# Split image into red, blue, and green channels
b, g, r = cv2.split(image)

# Upscale each color channel R, G, B using interpolation
def scaleHeight(arr, scale, method):
    height, width = arr.shape[0], arr.shape[1]
    scaled_height = int(height * scale)
    new_arr = np.zeros((scaled_height, width), dtype = np.uint8)

    x = np.arange(height)  
    for i in range(width):
        y = arr[:, i]
        if method == "Nearest":
            f = interpolate.interp1d(x, y, kind = "nearest")
        elif method == "Bilinear":
            f = interpolate.interp1d(x, y, kind = "linear")
        else:
            f = interpolate.CubicSpline(x, y, bc_type = "natural")
        h = np.linspace(0, height - 1, scaled_height) 
        new_arr[:, i] = np.clip(f(h), 0, 255).astype(np.uint8)

    return new_arr

def scaleWidth(arr, scale, method):
    height, width = arr.shape[0], arr.shape[1]
    scaled_width = int(width * scale)
    new_arr = np.zeros((height, scaled_width), dtype = np.uint8)

    x = np.arange(width)
    for i in range(height):
        y = arr[i] 
        if method == "Nearest":
            f = interpolate.interp1d(x, y, kind = "nearest")
        elif method == "Bilinear":
            f = interpolate.interp1d(x, y, kind = "linear")
        else:
            f = interpolate.CubicSpline(x, y, bc_type = "natural")
        h = np.linspace(0, width - 1, scaled_width)
        new_arr[i] = np.clip(f(h), 0, 255).astype(np.uint8)

    return new_arr

def upscale(channel, scale):
    return scaleHeight(scaleWidth(channel, scale, method), scale, method)

scaled_r = upscale(r, scale)
scaled_g = upscale(g, scale)
scaled_b = upscale(b, scale)

# Merge channels back into a color image
scaled = cv2.merge([scaled_b, scaled_g, scaled_r])

# Show the upscaled image
if method != "Nearest" and method != "Bilinear":
  method = "Bicubic"
if method == "Nearest":
  method = "Nearest Neighbour"
cv2.imshow(f"Scaled ({method})", scaled) 
cv2.waitKey(0)