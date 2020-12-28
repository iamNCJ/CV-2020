import argparse
import os

import cv2
import numpy as np


def pca(X, energy):
    _mean = np.mean(X, axis=0)
    _centered_data = X - _mean
    u, s, vh = np.linalg.svd(_centered_data)
    accumulator = np.cumsum(s / sum(s))
    n_pc = np.argwhere(accumulator > energy)[0][0] + 1
    _components = vh[:n_pc]
    _projected = u[:, :n_pc] * s[:n_pc]
    return _projected, _components, _mean, _centered_data


def reconstruction(Y, C, M, h, w, image_index):
    weights = np.dot(Y, C.T)
    centered_vector = np.dot(weights[image_index, :], C)
    recovered_image = (M + centered_vector).reshape(h, w)
    return recovered_image


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face trainer')
    arg_parser.add_argument('--energy', dest='energy', type=float, default=0.8)
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    arg_parser.add_argument('--data', dest='training_data', type=str, default='./data/JAFFE')
    arg_parser.add_argument('--resize', dest='size', type=int, default=64)
    args = arg_parser.parse_args()
    print(f'Start training with arguments {args}')

    training_photos = [args.training_data + '/' + photo for photo in os.listdir(args.training_data)]
    training_data = np.stack([cv2.resize(cv2.imread(image, cv2.COLOR_BGR2GRAY), (args.size, args.size)) for image in training_photos])
    n_samples, h, w = training_data.shape

    print('Calculating PCA')
    projected, components, mean, centered_data = pca(training_data.reshape(n_samples, h * w), energy=args.energy)
    print(f'Using {components.shape[0]} principal components')

    with open(args.model_file, 'wb') as f:
        np.save(f, projected)
        np.save(f, components)
        np.save(f, mean)
        np.save(f, centered_data)
    print(f'Saving model to {args.model_file}')

    # with open(args.model_file, 'rb') as f:
    #     a = np.load(f)
    #     b = np.load(f)
    # print(a)
