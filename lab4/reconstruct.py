import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot(images, counts, row, col):
    plt.figure()
    assert row * col >= counts
    for i in range(counts):
        plt.subplot(row, col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def reconstruction(image, pc, _mean, _size, n_pc):
    pc = pc[:][:n_pc]
    project_vector = image.dot(pc.T)
    centered_vector = np.dot(project_vector, pc)
    recovered_image = (_mean + centered_vector).reshape(_size, _size)
    return recovered_image


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face reconstructor')
    arg_parser.add_argument('--input', dest='input_image', type=str, required=True)
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    args = arg_parser.parse_args()

    # Reload model
    with open(args.model_file, 'rb') as f:
        size = np.load(f)
        projected = np.load(f)
        components = np.load(f)
        mean = np.load(f)
        centered_data = np.load(f)

    input_image = cv2.resize(cv2.imread(args.input_image, cv2.COLOR_BGR2GRAY), (size, size)).reshape(-1)
    rec_images = [reconstruction(input_image, components, mean, size, num_pc) for num_pc in (10, 25, 50, 100, 150, components.shape[0])]
    plot(rec_images, 6, 2, 3)
