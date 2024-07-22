import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


img = cv2.imread("Class_Tasks/Day_1/Images/IMG_1.jpeg")

img_down = cv2.pyrDown(img)

fig = plt.figure(figsize=(10, 10), dpi=72)
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img)
ax1.set_title("Original Image")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(img_down)
ax2.set_title("Downsampled Image")

img_up = cv2.pyrUp(img_down)
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(img_up)
ax3.set_title("Upsampled Image on Downsampled")

plt.show()

img_down = np.float32(img_down)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(
    img_down, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_down.shape))

fig = plt.figure(figsize=(10, 10), dpi=72)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img)
ax1.set_title("Original Image")

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(res2)
ax2.set_title("Quantized Image")
plt.show()

cv2.imwrite("Class_Tasks/Day_1/Outputs/quantized.jpg", res2)
