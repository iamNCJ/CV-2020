import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import plot, save_model


def pca(X, energy):
    _mean = np.mean(X, axis=0)
    _centered_data = X - _mean
    u, s, vh = np.linalg.svd(_centered_data)
    accumulator = np.cumsum(s / sum(s))
    n_pc = np.argwhere(accumulator >= energy)[0][0] + 1
    _components = vh[:n_pc]
    _components = _components / np.linalg.norm(_components, axis=1)[:, None]
    _projected = _centered_data.dot(_components.T)
    return _projected, _components, _mean, _centered_data


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face trainer')
    arg_parser.add_argument('--energy', dest='energy', type=float, default=1.0)
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    arg_parser.add_argument('--resize', dest='size', type=int, default=64)
    arg_parser.add_argument("--headless", dest="headless", action="store_true", default=False, required=False,
                            help="Train in headless mode")
    args = arg_parser.parse_args()
    print(f'Start training with arguments {args}')
    assert args.size > 0 and args.energy > 0

    training_photos = [f'./data/processed/{i}/{j}.png' for i in range(1, 42) for j in range(1, 6)]
    labels = np.array([i for i in range(1, 42) for _ in range(1, 6)])
    training_data = np.stack(
        [cv2.equalizeHist(cv2.resize(cv2.imread(image, cv2.COLOR_BGR2GRAY), (args.size, args.size))) for image in
         training_photos])

    n_samples, h, w = training_data.shape

    print('Calculating PCA')
    projected, components, mean, centered_data = pca(training_data.reshape(n_samples, h * w), energy=args.energy)
    print(f'Using {components.shape[0]} principal components')

    save_model(args.model_file, args.size, projected, components, mean, centered_data, labels)
    print(f'Saving model to {args.model_file}')

    if not args.headless:
        # Show mean face
        plt.imshow(mean.reshape((args.size, args.size)), cmap='gray')
        plt.show()
        # Show 10 pcs
        plot(components.reshape(components.shape[0], h, w), 10, 3, 4, [str(i) for i in range(1, 11)])
