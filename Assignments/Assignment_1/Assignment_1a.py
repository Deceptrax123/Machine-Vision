import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task1: Basic Image Statistics and Color Space Conversion


def task_1():
    img = cv2.imread("Lab_1/images/image_3.jpeg")

    # mean and Standard Deviation
    mean_blue, std_blue = np.mean(img[:, :, 0]), np.std(img[:, :, 0])
    mean_green, std_green = np.mean(img[:, :, 1]), np.std(img[:, :, 1])
    mean_red, std_red = np.mean(img[:, :, 2]), np.std(img[:, :, 2])

    print(f"Mean and Standard Deviation of Blue: {mean_blue},{std_blue}")
    print(f"Mean and Standard Deviation of Green: {mean_green},{std_green}")
    print(f"Mean and Standard Deviation of Red: {mean_red},{std_red}")

    # Histogram
    fig = plt.figure(figsize=(20, 20), dpi=72)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.hist(img[:, :, 0])
    ax1.set_title("Blue Histogram")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.hist(img[:, :, 1])
    ax2.set_title("Green Histogram")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.hist(img[:, :, 2])
    ax3.set_title("Red Histogram")
    plt.show()

    # Color Space Conversion
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title("BGR Space")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img_hsv)
    ax2.set_title("HSV Space")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img_lab)
    ax3.set_title("LAB Space")
    plt.show()

# Simple Image Segmentation using Thresholding


def task_2():
    img = cv2.imread("Lab_1/images/image_3.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Binary Thresholding
    _, threshold = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

    cv2.imshow("Original Image", img_gray)
    cv2.imshow("Segmented Image", threshold)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
# Color Based Segmentation


def task_3():
    img = cv2.imread("Lab_1/images/image_2.jpeg")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, (36, 25, 25), (70, 255, 255))
    image_mask = mask > 0
    segmented = np.zeros_like(img, np.uint8)
    segmented[image_mask] = img[image_mask]

    cv2.imshow("HSV Space", img_hsv)
    cv2.imshow("Original Image", img)
    cv2.imshow("Segmented Image", segmented)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    task_3()
