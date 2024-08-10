import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

# Introduction to Histogram Equalization


def task_1():
    img = cv2.imread("Lab_2/Images/image.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.hist(img_gray)
    plt.show()
    # Cumulative Distribiution Function
    norm_cdf = scipy.stats.norm.cdf(img_gray)
    print(norm_cdf)

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(norm_cdf*img_gray)  # Expectation
    ax1.set_title("Histogram Equalised")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_gray)
    ax2.set_title("Original Image")

    plt.show()

# Comparing Histogram Equalization Techniques


def task_2():
    img = cv2.imread("Lab_1/images/image_2.jpeg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)  # normal equalisation

    clahe = cv2.createCLAHE(clipLimit=5)  # clahe equalisation
    final_img_clahe = clahe.apply(img_gray)+30

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(img_gray)
    ax1.set_title("Gray Scale")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(equ)
    ax2.set_title("Histogram Equalised")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(final_img_clahe)
    ax3.set_title("CLAHE")

    plt.show()
# Implementing Histogram Equalization on Color Images


def task_3():
    img = cv2.imread("Lab_1/images/image_2.jpeg")

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    img_b_eq = cv2.equalizeHist(img_b)
    img_r_eq = cv2.equalizeHist(img_r)
    img_g_eq = cv2.equalizeHist(img_g)

    img_eq = np.stack([img_b_eq, img_r_eq, img_g_eq], axis=2)

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title("Original Image")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_eq)
    ax2.set_title("Histogram Color Channel Equalised")

    plt.show()


if __name__ == '__main__':
    task_3()
