import cv2
import numpy as np
import matplotlib.pyplot as plt
from block_distortion import distort_image

# Geometric rectifications


def tasks():
    img = cv2.imread("Assignments/Assignment_2/architecture.jpeg")

    # add distortion to the image
    distorted = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("Assignments/Assignment_2/distorted.jpeg", distorted)

    distorted = cv2.imread("Assignments/Assignment_2/distorted.jpeg")

    # Mark key points
    orb = cv2.ORB_create()

    kp = orb.detect(img, None)

    kp, des = orb.compute(img, kp)

    img2 = cv2.drawKeypoints(distorted, kp, None, color=(0, 0, 255), flags=0)

    # Bilinear Interpolation
    rectified = cv2.resize(distorted, dsize=(512, 512),
                           interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Distorted Image", distorted)
    cv2.imshow("GCP Marked", img2)
    cv2.imshow("Bilinear Interpolation", rectified)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tasks()
