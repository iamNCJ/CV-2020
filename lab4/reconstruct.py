import argparse

import cv2
import numpy as np

from utils import plot, load_model


def reconstruction(image, pc, _mean, _size, n_pc):
    pc = pc[:n_pc]
    project_vector = (image - _mean).dot(pc.T)
    centered_vector = np.dot(project_vector, pc)
    recovered_image = (_mean + centered_vector).reshape(_size, _size)
    return recovered_image


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face reconstructor')
    arg_parser.add_argument('--input', dest='input_image', type=str, required=False, default='./data/img.png')
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    args = arg_parser.parse_args()

    # Reload model
    size, projected, components, mean, centered_data, labels = load_model(args.model_file)

    # cv2.imshow('', cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2GRAY))
    # cv2.waitKey(-1)
    input_image = cv2.resize(cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2GRAY), (size, size)).reshape(-1)
    pcs = (10, 25, 50, 100, 150, 175, 200, components.shape[0])
    rec_images = [reconstruction(input_image, components, mean, size, num_pc) for num_pc in pcs]
    rec_images.append(input_image.reshape((size, size)))
    plot(rec_images, 9, 3, 3, [str(i) for i in pcs] + ['origin'])
