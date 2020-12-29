import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def pca(X, energy):
    _mean = np.mean(X, axis=0)
    _centered_data = X - _mean
    u, s, vh = np.linalg.svd(_centered_data)
    accumulator = np.cumsum(s / sum(s))
    n_pc = np.argwhere(accumulator >= energy)[0][0] + 1
    _components = vh[:n_pc]
    _projected = X.dot(_components.T)
    return _projected, _components, _mean, _centered_data


def plot(images, counts, row, col):
    plt.figure()
    assert row * col >= counts
    for i in range(counts):
        plt.subplot(row, col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face trainer')
    arg_parser.add_argument('--energy', dest='energy', type=float, default=0.8)
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    arg_parser.add_argument('--data', dest='training_data', type=str, default='./data/JAFFE')
    arg_parser.add_argument('--resize', dest='size', type=int, default=64)
    arg_parser.add_argument("--headless", dest="headless", action="store_true", default=False, required=False, help="Train in headless mode")
    args = arg_parser.parse_args()
    print(f'Start training with arguments {args}')
    assert args.size > 0 and args.energy > 0

    training_photos = [args.training_data + '/' + photo for photo in os.listdir(args.training_data)]
    training_data = np.stack([cv2.resize(cv2.imread(image, cv2.COLOR_BGR2GRAY), (args.size, args.size)) for image in training_photos])
    n_samples, h, w = training_data.shape

    print('Calculating PCA')
    projected, components, mean, centered_data = pca(training_data.reshape(n_samples, h * w), energy=args.energy)
    print(f'Using {components.shape[0]} principal components')

    with open(args.model_file, 'wb') as f:
        np.save(f, args.size)
        np.save(f, projected)
        np.save(f, components)
        np.save(f, mean)
        np.save(f, centered_data)
        np.save(f, training_data)
    print(f'Saving model to {args.model_file}')

    if not args.headless:
        # Show mean face
        plt.imshow(mean.reshape((args.size, args.size)), cmap='gray')
        plt.show()
        # Show 10 pcs
        plot(components.reshape(components.shape[0], h, w), 10, 3, 4)
