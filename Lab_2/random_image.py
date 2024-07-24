import cv2 
import numpy as np 
import matplotlib.pyplot as plt

random_img=np.random.randint(low=0,high=255,size=(64,64,3),dtype=np.uint8)
plt.imshow(random_img)
plt.show()