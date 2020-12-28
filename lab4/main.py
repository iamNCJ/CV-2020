import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

"""It helps visualising the portraits from the dataset."""
def plot_portraits(images, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def pca(X, n_pc):
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:, :n_pc] * S[:n_pc]
    return projected, components, mean, centered_data


def reconstruction(Y, C, M, h, w, image_index):
    weights = np.dot(Y, C.T)
    centered_vector = np.dot(weights[image_index, :], C)
    recovered_image = (M + centered_vector).reshape(h, w)
    return recovered_image


if __name__ == '__main__':
    dir = 'data/JAFFE'
    celebrity_photos = os.listdir(dir)[1:1001]
    celebrity_images = [dir + '/' + photo for photo in celebrity_photos]
    images = np.stack([cv2.resize(cv2.imread(image, cv2.COLOR_BGR2GRAY), (64, 64)) for image in celebrity_images])
    n_samples, h, w = images.shape
    plot_portraits(images, h, w, n_row=4, n_col=4)
    mean = np.mean(images, axis=0)
    plt.imshow(mean, cmap='gray')
    plt.show()

    n_components = 50
    X = images.reshape(n_samples, h * w)
    P, C, M, Y = pca(X, n_pc=n_components)
    eigenfaces = C.reshape((n_components, h, w))
    plot_portraits(eigenfaces, h, w, n_row=4, n_col=4)

    recovered_images = [reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
    plot_portraits(recovered_images, h, w, n_row=4, n_col=4)
