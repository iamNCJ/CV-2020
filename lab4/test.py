import argparse

import cv2
import numpy as np

from utils import plot, load_model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face tester')
    arg_parser.add_argument('--input', dest='input_image', type=str, required=False, default='./data/img.png')
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    args = arg_parser.parse_args()

    # Reload model
    size, projected, components, mean, centered_data, labels = load_model(args.model_file)

    input_image = cv2.resize(cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2GRAY), (size, size)).reshape(-1)
    project_vector = (input_image - mean).dot(components.T)
    distances = np.sum((projected - project_vector) ** 2, axis=1)
    idx = np.argmin(distances)
    nearest_img = (centered_data[idx] + mean).reshape(size, size)
    detected_img = (nearest_img + input_image.reshape(size, size)) / 2
    plot([input_image.reshape(size, size), nearest_img, detected_img], 3, 1, 3)
