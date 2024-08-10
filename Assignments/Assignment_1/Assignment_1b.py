import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Image Negative Transformation


def task_1():
    img = cv2.imread("Lab_1/images/image_3.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_negative = (256-1)-img_gray

    cv2.imshow("Image", img)
    cv2.imshow("Gray Scale Image", img_gray)
    cv2.imshow("Negative Transform", img_negative)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Gamma Correction


def task_2():
    img = cv2.imread("Lab_1/images/image_3.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gamma = 0.2*(img_gray**0.5)
    img_gamma1 = 0.2*(img_gray**1)
    img_gamma2 = 0.2*(img_gray**2)

    cv2.imshow("Image", img)
    cv2.imshow("Gray Scale", img_gray)
    cv2.imshow("Gamma = 0.5", img_gamma)
    cv2.imshow("Gamma = 1", img_gamma1)
    cv2.imshow("Gamma = 2", img_gamma2)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Log Transform


def task_3():
    img = cv2.imread("Lab_1/images/image_3.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_log = 0.2*np.log1p(img_gray)
    img_log = cv2.imwrite(
        "Assignments/Assignment_1/log_transform.jpg", img_log)
    img_log = cv2.imread("Assignments/Assignment_1/log_transform.jpg")

    cv2.imshow("Image", img)
    cv2.imshow("Gray Scale", img_gray)
    cv2.imshow("Log Transformed Image", img_log)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Compare Transforms


def task_4():
    img = cv2.imread("Lab_1/images/image_3.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_negative = (256-1)-img_gray
    img_gamma = 0.2*(img_gray**2)
    img_log = 0.2*np.log1p(img_gray)
    cv2.imwrite("Assignments/Assignment_1/Log_Transform.jpg", img_log)

    img_log = cv2.imread("Assignments/Assignment_1/Log_transform.jpg")

    cv2.imshow("Image", img)
    cv2.imshow("Gray Scale", img_gray)
    cv2.imshow("Negative Transform", img_negative)
    cv2.imshow("Gamma = 2", img_gamma)
    cv2.imshow("Log Transformed Image", img_log)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# Apply Transformations to Color Images


def task_5():
    img = cv2.imread("Lab_1/images/image_3.jpeg")

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    def log_trans(img):
        return np.log1p(img)

    def negative(img):
        return (256-1)-img

    def gamma(img):
        return 0.2*(img**0.5)

    img_b_log = log_trans(img_b)
    img_b_negative = negative(img_b)
    img_b_gamma = gamma(img_b)

    img_g_log = log_trans(img_g)
    img_g_negative = negative(img_g)
    img_g_gamma = gamma(img_g)

    img_r_log = log_trans(img_r)
    img_r_negative = negative(img_r)
    img_r_gamma = gamma(img_r)

    log_image = np.stack((img_b_log, img_g_log, img_r_log), axis=2)
    cv2.imwrite("Assignments/Assignment_1/log_transform.jpeg", log_image)

    log_image = cv2.imread("Assignments/Assignment_1/log_transform.jpg")
    negative_image = np.stack(
        (img_b_negative, img_g_negative, img_r_negative), axis=2)
    gamma_image = np.stack((img_b_gamma, img_g_gamma, img_r_gamma), axis=2)

    cv2.imshow("Image", img)
    cv2.imshow("Log Transformed Image", log_image)
    cv2.imshow("Negative Transformed Image", negative_image)
    cv2.imshow("Gamma Transformed Image", gamma_image)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    exit = -1
    while exit != 1:
        opt = int(input("Enter Task Number: "))
        if opt == 1:
            task_1()
        elif opt == 2:
            task_2()
        elif opt == 3:
            task_3()
        elif opt == 4:
            task_4()
        else:
            task_5()

        exit = int(input("Press 1 to exit, 0 to continue: "))
