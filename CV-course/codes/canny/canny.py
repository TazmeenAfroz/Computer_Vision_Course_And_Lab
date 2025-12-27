import cv2
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt

# ---------- Step 1: Convolution ----------
def convolution2d(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    padding_h = (kernel_h - 1) // 2
    padding_w = (kernel_w - 1) // 2

    padded_image = np.pad(image, ((padding_h, padding_h), (padding_w, padding_w)), mode='constant', constant_values=0)
    output = np.zeros((image_h, image_w))

    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)
    return output

# ---------- Step 2: Gaussian Filter ----------
def gaussian_filter2d(sigma):
    size = int(2 * (np.ceil(3 * sigma)) + 1)  # size based on 3σ rule
    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            x_val = x - size // 2
            y_val = y - size // 2
            kernel[x, y] = (1/(2 * np.pi * sigma**2)) * np.exp(-(x_val**2 + y_val**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

# ---------- Step 3: Gradient (Sobel) ----------
def gradient(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = convolution2d(image, sobel_x)
    Gy = convolution2d(image, sobel_y)

    mag = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle[angle < 0] += 180  # normalize 0–180
    return mag, angle

# ---------- Step 4: Non-Maximum Suppression ----------
def non_max_suppression(mag, angle):
    h, w = mag.shape
    Z = np.zeros((h, w), dtype=np.float32)
    angle = angle % 180

    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if (mag[i,j] >= q) and (mag[i,j] >= r):
                Z[i,j] = mag[i,j]
            else:
                Z[i,j] = 0
    return Z

# ---------- Step 5: Double Threshold ----------
def double_threshold(img, low, high):
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res = np.zeros_like(img, dtype=np.uint8)
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

# ---------- Step 6: Hysteresis ----------
def hysteresis(img, weak, strong=255):
    h, w = img.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

# ---------- Main Canny Function ----------
def canny(image, sigma=1, low=50, high=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gauss = gaussian_filter2d(sigma)
    smoothed = convolution2d(gray, gauss)

    mag, angle = gradient(smoothed)
    nonmax = non_max_suppression(mag, angle)
    dt, weak, strong = double_threshold(nonmax, low, high)
    final = hysteresis(dt, weak, strong)

    return final
