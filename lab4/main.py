# Press the green button in the gutter to run the script.
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir = 'data/JAFFE'
    celebrity_photos = os.listdir(dir)[1:1001]
    celebrity_images = [dir + '/' + photo for photo in celebrity_photos]
    images = []
    for image in celebrity_images:
        temp = cv2.imread(image, cv2.COLOR_BGR2GRAY)
        # temp.reshape(-1)
        images.append(temp)
    mean = np.mean(images, axis=0)
    plt.imshow(mean, cmap='gray')
    plt.show()
    pass
