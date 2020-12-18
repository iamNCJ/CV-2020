import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

cam = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    _, img = cam.read()
    # img = cv2.imread('assets/sample.jpg')

    # Color to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Spatial derivative calculation
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Structure tensor setup
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    kernel = np.ones((3, 3), np.float32)
    Sum_xx = cv2.filter2D(Ixx, -1, kernel)
    Sum_xy = cv2.filter2D(Ixy, -1, kernel)
    Sum_yy = cv2.filter2D(Iyy, -1, kernel)

    # Harris response calculation
    k = 0.04
    R = Sum_xx * Sum_yy - Sum_xy ** 2 - k * (Sum_xx + Sum_yy) ** 2

    dst = R
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    tmp = img.copy()
    tmp[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Display the resulting frame
    cv2.imshow('frame', tmp)

    if cv2.waitKey(1) == 32:
        h, w = gray.shape
        temp = np.zeros((h, w, 2, 2))
        temp[:, :, 0, 0] = Sum_xx
        temp[:, :, 0, 1] = Sum_xy
        temp[:, :, 1, 0] = Sum_xy
        temp[:, :, 1, 1] = Sum_yy

        eigen, _ = np.linalg.eig(temp)
        lambda1 = eigen[:, :, 0]
        lambda2 = eigen[:, :, 1]

        lambda_max = np.maximum(lambda1, lambda2)
        lambda_min = np.minimum(lambda1, lambda2)

        # Non-maximum suppression
        pos = np.argwhere(R > 0.01 * R.max())
        for a, b in tqdm(pos):
            x0 = max(0, a - 1)
            x1 = min(h, a + 1)
            y0 = max(0, b - 1)
            y1 = min(w, b + 1)
            if R[a, b] == np.max(R[x0:x1, y0:y1]):
                cv2.drawMarker(img, (b, a), (0, 0, 255))

        fig, subplots = plt.subplots(2, 2)
        subplots[0, 0].imshow(lambda_max, cmap='hot', interpolation='nearest')
        subplots[0, 0].set_title(r'$\lambda_{max}$')
        subplots[0, 0].axis('off')
        subplots[1, 0].set_title(r'$\lambda_{min}$')
        subplots[1, 0].imshow(lambda_min, cmap='hot', interpolation='nearest')
        subplots[1, 0].axis('off')
        subplots[0, 1].imshow(R, cmap='hot', interpolation='nearest')
        subplots[0, 1].set_title(r"$R$")
        subplots[0, 1].axis('off')
        subplots[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        subplots[1, 1].set_title("Result")
        subplots[1, 1].axis('off')

        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', img)

        flag = False
        while True:
            char = cv2.waitKey(-1)
            if char == 32:
                break
            elif char == ord('s'):
                fig.savefig('harris.png')
            elif char == ord('q'):
                flag = True
                break
        if flag:
            break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
