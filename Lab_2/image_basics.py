import cv2 
import matplotlib.pyplot as plt 
import numpy as np 


#Black and white
img =cv2.imread("Lab_2/Images/black_white.jpeg")

plt.imshow(img)
plt.show()
print(img)

print("Maximum: ",np.max(img))
print("Minimum: ",np.min(img))

#Grayscale
img =cv2.imread("Lab_2/Images/gray_scale_image.jpeg")

plt.imshow(img)
plt.show()
print(img)

print("Maximum: ",np.max(img))
print("Minimum: ",np.min(img))

#BGR
img =cv2.imread("Lab_2/Images/image.jpeg")

plt.imshow(img)
plt.show()
print(img)

print("Maximum: ",np.max(img))
print("Minimum: ",np.min(img))