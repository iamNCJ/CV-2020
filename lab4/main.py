import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

def pca(X, n_pc):
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:, :n_pc] * S[:n_pc]

    return projected, components, mean, centered_data


if __name__ == '__main__':
    dir = 'data/JAFFE'
    celebrity_photos = os.listdir(dir)[1:1001]
    celebrity_images = [dir + '/' + photo for photo in celebrity_photos]
    images = np.stack([cv2.imread(image, cv2.COLOR_BGR2GRAY) for  image in celebrity_images])
    mean = np.mean(images, axis=0)
    plt.imshow(mean, cmap='gray')
    plt.show()

    n_samples, h, w = images.shape
    n_components = 50
    X = images.reshape(n_samples, h * w)
    P, C, M, Y = pca(X, n_pc=n_components)
    eigenfaces = C.reshape((n_components, h, w))
    for face in tqdm(eigenfaces):
        plt.imshow(face, cmap='gray')
        plt.show()
    # eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    # plot_portraits(eigenfaces, eigenface_titles, h, w, 4, 4)
