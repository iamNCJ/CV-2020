import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)

while True:
    start = time.time()
    # Capture frame-by-frame
    _, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    stop = time.time()
    fps = 1 / (stop - start)
    frame = cv2.putText(img, f'{int(fps)}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()