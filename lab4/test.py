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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face tester')
    arg_parser.add_argument('--input', dest='input_image', type=str, required=False, default='./data/img.png')
    arg_parser.add_argument('--model', dest='model_file', type=str, default='model.npy')
    args = arg_parser.parse_args()

    # Reload model
    with open(args.model_file, 'rb') as f:
        size = np.load(f)
        projected = np.load(f)
        components = np.load(f)
        mean = np.load(f)
        centered_data = np.load(f)

    input_image = cv2.resize(cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2GRAY), (size, size)).reshape(-1)
    project_vector = (input_image - mean).dot(components.T)
    distances = np.sum((projected - project_vector) ** 2, axis=1)
    idx = np.argmin(distances)
    nearest_img = (centered_data[idx] + mean).reshape(size, size)
    detected_img = (nearest_img + input_image.reshape(size, size)) / 2
    plot([input_image.reshape(size, size), nearest_img, detected_img], 3, 1, 3)
