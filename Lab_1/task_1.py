import cv2
import numpy as np
import matplotlib.pyplot as plt

# get images
img1 = cv2.imread("Lab_1/images/images.jpeg")
img2 = cv2.imread("Lab_1/images/image_2.jpeg")
img3 = cv2.imread("Lab_1/images/image_3.jpeg")

# Convert ro gray scale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Display the images
fig = plt.figure(figsize=(10, 10), dpi=72)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img1)

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img1_gray)

plt.show()

fig = plt.figure(figsize=(10, 10), dpi=72)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img2)

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img2_gray)
plt.show()

fig = plt.figure(figsize=(10, 10), dpi=72)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img3)

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img3_gray)
plt.show()

# Resize
img1_gray = cv2.resize(img1_gray, (250, 250))
img2_gray = cv2.resize(img2_gray, (250, 250))
img3_gray = cv2.resize(img3_gray, (250, 250))

# Create a collage in image
collage = np.hstack([img1_gray, img2_gray, img3_gray])

plt.imshow(collage)
plt.show()
