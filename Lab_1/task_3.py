import cv2
import numpy as np
import matplotlib.pyplot as plt

def user_convert(img, option):
    if option == 'rgb':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()
        
    elif option == 'hsv':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        plt.imshow(img_hsv)
        plt.show()


if __name__=='__main__':
    img = cv2.imread("Lab_1/images/images.jpeg")

    plt.imshow(img)
    plt.title("Interactive Plot showing Pixels and Coordinates")
    plt.show()

    while True:
        op=int(input("Enter option value: "))
        
        if op==0:
            break
        elif op==1:
            user_convert(img,'rgb')
        elif op==2:
            user_convert(img,'hsv')
    