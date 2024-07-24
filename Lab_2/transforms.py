import cv2 
import matplotlib.pyplot as plt
import numpy as np 

img=cv2.imread("Lab_2/Images/image.jpeg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_linear=(256-1)-img_gray
img_log=0.2*np.log1p(img_gray)
img_log_inv=0.2*np.exp(0.01*img_gray)
img_gamma=0.2*(img_gray**2.8)

fig = plt.figure(figsize=(10, 20), dpi=72)
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img)
ax1.set_title("BGR")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(img_gray)
ax2.set_title("Gray Scale")

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(img_linear)
ax3.set_title("Linear")

ax4 = fig.add_subplot(2, 3, 1)
ax4.imshow(img_log)
ax4.set_title("Logarthimic")


ax5 = fig.add_subplot(2, 3, 2)
ax5.imshow(img_gamma)
ax5.set_title("Gamma")

ax6 = fig.add_subplot(2, 3, 3)
ax6.imshow(img_log_inv)
ax6.set_title("Inverse Log")
plt.show()

#Gamma Analysis
img_gamma=0.2*(img_gray**2.8)
img_gamma1=0.2*(img_gray**2)
img_gamma2=0.2*(img_gray**1.5)
img_gamma3=0.2*(img_gray**0.7)
img_gamma4=0.2*(img_gray**0.2)
img_gamma5=0.2*(img_gray**3.2)


fig = plt.figure(figsize=(12, 12), dpi=72)
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img_gamma)
ax1.set_title("Gamma=2.8")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(img_gamma1)
ax2.set_title("Gamma=2")

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(img_gamma2)
ax3.set_title("Gamma=1.5")

ax4 = fig.add_subplot(2, 3, 1)
ax4.imshow(img_gamma3)
ax4.set_title("Gamma=0.7")

ax5 = fig.add_subplot(2, 3, 2)
ax5.imshow(img_gamma4)
ax5.set_title("Gamma=0.2")

ax6 = fig.add_subplot(2, 3, 3)
ax6.imshow(img_gamma5)
ax6.set_title("Gamma=3.2")

plt.show()
