import matplotlib.pyplot as plt
import numpy as np


def plot(images, counts, row, col):
    plt.figure()
    assert row * col >= counts
    for i in range(counts):
        plt.subplot(row, col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def load_model(model_file):
    with open(model_file, 'rb') as f:
        size = np.load(f)
        projected = np.load(f)
        components = np.load(f)
        mean = np.load(f)
        centered_data = np.load(f)
        labels = np.load(f)
        return size, projected, components, mean, centered_data, labels


def save_model(model_file, size, projected, components, mean, centered_data, labels):
    with open(model_file, 'wb') as f:
        np.save(f, size)
        np.save(f, projected)
        np.save(f, components)
        np.save(f, mean)
        np.save(f, centered_data)
        np.save(f, labels)
