import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(1)
cnt = 0
while True:
    # Capture frame-by-frame
    _, img = cam.read()
    # img = cv2.imread('assets/sample.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    kernel = np.ones((3, 3), np.float32)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)

    k = 0.04
    R = Sxx * Syy - Sxy ** 2 - k * (Sxx + Syy) ** 2

    dst = R
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Display the resulting frame
    cv2.imshow('frame', img)

    if cv2.waitKey(1) == 32:
        h, w = gray.shape
        temp = np.zeros((h, w, 2, 2))
        temp[:, :, 0, 0] = Sxx
        temp[:, :, 0, 1] = Sxy
        temp[:, :, 1, 0] = Sxy
        temp[:, :, 1, 1] = Syy

        w, _ = np.linalg.eig(temp)
        lambda1 = w[:, :, 0]
        lambda2 = w[:, :, 1]

        lambda_max = np.maximum(lambda1, lambda2)
        lambda_min = np.minimum(lambda1, lambda2)

        _, fig = plt.subplots(2, 2)

        fig[0, 0].imshow(lambda_max, cmap='hot', interpolation='nearest')
        fig[0, 0].set_title(r'$\lambda_{max}$')
        fig[0, 0].axis('off')
        fig[1, 0].set_title(r'$\lambda_{min}$')
        fig[1, 0].imshow(lambda_min, cmap='hot', interpolation='nearest')
        fig[1, 0].axis('off')
        fig[0, 1].imshow(R, cmap='hot', interpolation='nearest')
        fig[0, 1].set_title(r"$R$")
        fig[0, 1].axis('off')
        fig[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        fig[1, 1].set_title("Result")
        fig[1, 1].axis('off')

        plt.show()

        while cv2.waitKey(-1) != 32:
            pass

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
