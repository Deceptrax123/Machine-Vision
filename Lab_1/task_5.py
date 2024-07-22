import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img=cv2.imread("Lab_1/images/images.jpeg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_map=cv2.applyColorMap(img_gray,colormap=cv2.COLORMAP_OCEAN)
img_map2=cv2.applyColorMap(img_gray,colormap=cv2.COLORMAP_WINTER)

fig = plt.figure(figsize=(10, 10), dpi=72)
ax1 = fig.add_subplot(1, 4, 1)
ax1.imshow(img)
ax1.set_title("BGR")

ax2 = fig.add_subplot(1, 4, 2)
ax2.imshow(img_gray)
ax2.set_title("Gray Scale")

ax3 = fig.add_subplot(1, 4, 3)
ax3.imshow(img_map)
ax3.set_title("Color Map Ocean")

ax4 = fig.add_subplot(1, 4, 4)
ax4.imshow(img_map2)
ax4.set_title("Color Map Winter")

plt.show()
