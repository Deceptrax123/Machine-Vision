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
    ax2.set_title("Color Channel Equalised")

    plt.show()


def task_4():

    def equalisation(img, app):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gray)

        fig = plt.figure(figsize=(10, 10), dpi=72)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.set_title(f"Original {app} Image")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img_eq)
        ax2.set_title(f"Equalised for {app}")
        plt.show()

    equalisation(cv2.imread("Code/Assets/breast_cancer.tif"), 'Medical')
    equalisation(cv2.imread("Code/Assets/handwriting.jpg"),
                 'Scanned Documents')
    equalisation(cv2.imread('Code/Assets/satellite.tif'), 'Satellite')
    equalisation(cv2.imread("Code/Assets/night_vision.jpg"), "Night Vision")

# Histogram Equalization for Low Contrast Images


def task_5():
    low_contrast_img = cv2.imread(
        "/Users/smudge/Desktop/Code/Assets/low_contrast_image.jpeg")
    img_gray = cv2.cvtColor(low_contrast_img, cv2.COLOR_BGR2GRAY)

    img_eq = cv2.equalizeHist(img_gray)

    cv2.imshow("Low Contrast Image", low_contrast_img)
    cv2.imshow("Equalised", img_eq)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Multi-Scale Histogram Equalization


def task_6():
    img = cv2.imread("Lab_2/Images/image.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_re1 = cv2.resize(img_gray, (img.shape[0]//2, img.shape[1]//2))
    img_re2 = cv2.resize(img_gray, (img.shape[0]*2, img.shape[1]*2))

    img_re1_eq = cv2.equalizeHist(img_re1)
    img_re2_eq = cv2.equalizeHist(img_re2)

    cv2.imshow("Original Image", img)
    cv2.imshow("Resized by 50%", img_re1)
    cv2.imshow("Resized by 200%", img_re2)
    cv2.imshow("50% Image Equalised", img_re1_eq)
    cv2.imshow("200% Image Equalised", img_re2_eq)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Histogram Equalization for Image Enhancement


def task_7():
    img1 = cv2.imread("Lab_2/Images/image.jpeg")
    img2 = cv2.imread(
        "/Users/smudge/Desktop/Code/Assets/low_contrast_image.jpeg")
    img3 = cv2.imread("/Users/smudge/Desktop/Code/Assets/low_detail.jpeg")

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    img_eq1 = cv2.equalizeHist(img1_gray)
    img_eq2 = cv2.equalizeHist(img2_gray)
    img_eq3 = cv2.equalizeHist(img3_gray)

    cv2.imshow("Detailed Image", img1)
    cv2.imshow("Low Contrast Image", img2)
    cv2.imshow("Low Detail Image", img3)
    cv2.imshow("Detailed Image Equalised", img_eq1)
    cv2.imshow("Low Contrast Image Equalised", img_eq2)
    cv2.imshow("Low Detail Image Equalised", img_eq3)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Histogram Equalization in Image Segmentation


def task_8():
    img = cv2.imread("Lab_1/images/image_2.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

    # histogram equalised segmentation
    img_eq = cv2.equalizeHist(img_gray)
    _, threshold_eq = cv2.threshold(img_eq, 120, 255, cv2.THRESH_BINARY)

    cv2.imshow("Original Image", img_gray)
    cv2.imshow("Segmented Image without Histogram Equalisation", threshold)
    cv2.imshow("Segmented Image with Histogram Equalisation", threshold_eq)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    task_8()
