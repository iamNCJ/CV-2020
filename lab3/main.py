import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
cnt = 0
# fps = 60

# start = time.time()
while True:
    # cnt += 1
    # Capture frame-by-frame
    _, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    h, w = gray.shape
    kernel = np.ones((3, 3), np.float32)
    temp = np.zeros((h, w, 2, 2))
    temp[:, :, 0, 0] = cv2.filter2D(Ixx, -1, kernel)
    temp[:, :, 0, 1] = cv2.filter2D(Ixy, -1, kernel)
    temp[:, :, 1, 0] = cv2.filter2D(Ixy, -1, kernel)
    temp[:, :, 1, 1] = cv2.filter2D(Iyy, -1, kernel)

    w, _ = np.linalg.eig(temp)
    lambda1 = w[:, :, 0]
    lambda2 = w[:, :, 1]

    # plt.imshow(lambda1, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(lambda2, cmap='hot', interpolation='nearest')
    # plt.show()

    # k = 0.1
    # R = lambda1 * lambda2 - k * (lambda1 + lambda2) ** 2
    #
    # plt.imshow(R, cmap='hot', interpolation='nearest')
    # plt.show()

    # lambda_max = np.maximum(lambda1, lambda2)
    lambda_min = np.minimum(lambda1, lambda2)

    # plt.imshow(lambda_max, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(lambda_min, cmap='hot', interpolation='nearest')
    # plt.show()

    dst = lambda_min

    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.1 * dst.max()] = [0, 0, 255]

    # if cnt % 5 == 0:
    #     fps = 5 / (time.time() - start)
    #     start = time.time()
    # img = cv2.putText(img, f'{int(fps)}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=2)

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()