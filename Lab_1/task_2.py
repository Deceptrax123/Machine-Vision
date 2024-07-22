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


def convert(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title("BGR Space")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img_bgr)
    ax2.set_title("RGB Space")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img_hsv)
    ax3.set_title("HSV Space")

    plt.show()


if __name__ == '__main__':
    img = cv2.imread("Lab_1/images/images.jpeg")
    convert(img)

    opt = input("Enter rgb or hsv as final output: ")
    user_convert(img, opt)
