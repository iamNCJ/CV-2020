import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My eigen-face batch tester')
    arg_parser.add_argument('--model', dest='model_file', type=str, default='new.npy')
    args = arg_parser.parse_args()

    # Reload model
    size, projected, components, mean, centered_data, labels = load_model(args.model_file)

    test_photos = [f'./data/processed/{i}/{j}.png' for i in range(1, 42) for j in range(6, 11)]
    dest_labels = np.array([i for i in range(1, 42) for _ in range(6, 11)])

    fig = plt.figure()
    res = []

    for n_pc in tqdm(range(1, len(dest_labels) + 1)):
        suc_count = 0
        _components = components[:n_pc]
        _projected = projected[:, :n_pc]
        for test_photo, dest_label in zip(test_photos, dest_labels):
            test_data = cv2.equalizeHist(cv2.resize(cv2.imread(test_photo, cv2.COLOR_BGR2GRAY), (size, size))).reshape(-1)
            project_vector = (test_data - mean).dot(_components.T)
            distances = np.sum((_projected - project_vector) ** 2, axis=1)
            idx = np.argmin(distances)
            if labels[idx] == dest_label:
                suc_count += 1

        res.append(suc_count / len(dest_labels))

    plt.plot(res)
    print(res)
    plt.show()