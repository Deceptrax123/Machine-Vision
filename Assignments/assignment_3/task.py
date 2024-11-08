import cv2
import matplotlib.pyplot as plt
import numpy as np


def task_1():
    img = cv2.imread("Assignments/Assignment_3/Assets/satellite.jpeg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, t1 = cv2.threshold(img_gray, 100, 200, cv2.THRESH_BINARY)
    _, t2 = cv2.threshold(img_gray, 100, 200, cv2.THRESH_OTSU)
    _, t3 = cv2.threshold(img_gray, 100, 200, cv2.THRESH_BINARY_INV)

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(t1)
    ax1.set_title("Binary Thresholding")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(t2)
    ax2.set_title("Otsu Thresholding")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(t3)
    ax3.set_title("Inverse Binary Thresholding")
    plt.show()

    # Contrast Stretching
    rmin = np.min(img_gray)
    rmax = np.max(img_gray)
    img_stretched = 255*((img_gray-rmin)/(rmax-rmin))

    plt.imshow(img_stretched)
    plt.title("Contrast Stretched Image")
    plt.show()

    # Apply Gamma Correction
    img_gray_corrected = 0.7*(img_gray**0.9)
    rmin = np.min(img_gray_corrected)
    rmax = np.max(img_gray_corrected)
    img_stretched = 255*((img_gray_corrected-rmin)/(rmax-rmin))

    plt.imshow(img_stretched)
    plt.title("Contrast Stretched Image after Gamma Correction")
    plt.show()


def task_2():
    img = cv2.imread("Assignments/Assignment_3/Assets/satellite.jpeg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram
    plt.hist(img_gray)
    plt.title("Image Histogram")
    plt.show()

    # Histogram Equalisation
    equ = cv2.equalizeHist(img_gray)

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(img_gray)
    ax1.set_title("Histogram before Equalisation")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(equ)
    ax2.set_title("Histogram after Equalisation")
    plt.show()

    # Adaptive Histogram Equalisation
    ada = cv2.createCLAHE(clipLimit=5)
    final_img = ada.apply(img_gray)+30

    fig = plt.figure(figsize=(10, 10), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(equ)
    ax1.set_title("Standard Equalisation")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(final_img)
    ax2.set_title("Adaptive Equalisation")
    plt.show()


def task_3():
    img = cv2.imread("Assignments/Assignment_3/Assets/satellite.jpeg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Image')
    plt.show()

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    image_with_sift = cv2.drawKeypoints(img, keypoints, None)
    plt.imshow(cv2.cvtColor(image_with_sift, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Features')
    plt.show()


def scene_cut_detect():
    cap = cv2.VideoCapture('Assignments/Assignment_3/Assets/video.mp4')
    # cut scene detection
    idx = 0
    history = list()

    prev_gray_scale = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if frame is None:
            break
        # gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        pixel_diff = frame-prev_gray_scale
        prev_gray_scale = frame
        idx += 1

        history.append(np.mean(pixel_diff))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plt.plot(history)
    plt.xlabel("Time")
    plt.ylabel("Pixel wise Differences")
    plt.show()


def video_segmentation():
    cap = cv2.VideoCapture('Assignments/Assignment_3/Assets/video.mp4')

    while (cap.isOpened()):
        # Extract Individual frames from the video
        ret, frame = cap.read()

        if frame is None:
            break
        # Spatio Temporal Segmentation
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray_scale, 120, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        cv2.imshow('Segmented', thresh)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


scene_cut_detect()
