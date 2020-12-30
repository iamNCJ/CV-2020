import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pathlib

def rotate_image(image, center, angle):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def process_single_image(img_file, label_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    with open(label_file, 'r') as f:
        left_y, left_x = f.readline().split(',')
        left_x, left_y = int(left_x), int(left_y)
        right_y, right_x = f.readline().split(',')
        right_x, right_y = int(right_x), int(right_y)
    angle = np.arctan((right_y - left_y) / (right_x - left_x)) * 180 / np.pi
    center = ((left_x + right_x) / 2, (left_y + right_y) / 2)
    distance = ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5
    ratio = 32 / distance
    img = cv2.resize(img, (int(h * ratio), int(w * ratio)))
    center = (int(center[0] * ratio), int(center[1] * ratio))
    img = rotate_image(img, center, angle)
    img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_REPLICATE)
    img = img[center[1] + 10:center[1] + 74, center[0]: center[0] + 64]
    # plt.imshow(img, cmap='gray')
    return img


if __name__ == '__main__':
    for i in tqdm(range(1, 41)):
        pathlib.Path(f'./data/processed/{i}').mkdir(parents=True, exist_ok=True)
        for j in range(1, 11):
            file = f'./data/att_faces_with_eyes/s{i}/{j}'
            img = process_single_image(file + '.pgm', file + '.txt')
            print(f'./data/processed/{i}/{j}.png')
            cv2.imwrite(f'./data/processed/{i}/{j}.png', img)
            # plt.imshow(img, cmap='gray')
            plt.show()
