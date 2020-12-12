import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Circle


def line_detection(img, num_rhos=360, num_thetas=360, threshold=0.5, min_count=0):
    print("Preprocessing image...")
    edge_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.bilateralFilter(edge_image, 90, 75, 150)
    edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Quantify parameter space
    height, width = edge_image.shape[:2]
    diagonal = np.sqrt(np.square(height) + np.square(width))
    # theta: 0~360, rho: 0~d/2, thus making the axis balanced
    delta_theta = 360 / num_thetas
    delta_rho = diagonal / 2 / num_rhos
    thetas = np.arange(0, 360, step=delta_theta)
    rhos = np.arange(0, diagonal / 2, step=delta_rho)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # Filling
    # normalization
    edge_points = np.argwhere(edge_image != 0) - np.array([[height / 2, width / 2]])
    # rho = x * cos(theta) + y * sin(theta)
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    # filling the accumulator with np.histogram2d()
    accumulator, _, _ = np.histogram2d(np.tile(thetas, rho_values.shape[0]), rho_values.reshape(-1), bins=[thetas, rhos])
    auto_threshold = np.max([min_count, np.max(accumulator) * threshold])

    # Results
    # pick up parameters which counts are large enough
    lines = np.argwhere(np.transpose(accumulator) > auto_threshold)
    rho_idx, theta_idx = lines[:, 0], lines[:, 1]
    rho, theta = rhos[rho_idx], thetas[theta_idx]

    # Visualization
    # 4 subplots
    print("Plotting in hough space...")
    _, fig = plt.subplots(2, 2)
    fig[0, 0].imshow(img)
    fig[0, 0].set_title("Original Image")
    fig[0, 0].axis('off')
    fig[0, 1].imshow(edge_image, cmap="gray")
    fig[0, 1].set_title("Edge Image")
    fig[0, 1].axis('off')
    fig[1, 0].set_facecolor((0, 0, 0))
    fig[1, 0].set_title("Hough Space")
    fig[1, 0].invert_yaxis()
    fig[1, 1].imshow(img)
    fig[1, 1].set_title("Detected Lines")
    fig[1, 1].axis('off')
    # plot parameters
    for _rho in tqdm(rho_values):
        fig[1, 0].plot(thetas, _rho, color="white", alpha=0.01)
    # plot selected parameters
    fig[1, 0].plot([theta], [rho], marker='o', color='yellow')
    # plot lines
    for _rho, _theta in zip(rho, theta):
        cos_theta = np.cos(np.deg2rad(_theta))
        sin_theta = np.sin(np.deg2rad(_theta))
        # calculate [x1, y1] and [x2, y2] from parametric functions
        x0 = (cos_theta * _rho) + width / 2
        y0 = (sin_theta * _rho) + height / 2
        x1 = int(x0 - diagonal * sin_theta)
        y1 = int(y0 + diagonal * cos_theta)
        x2 = int(x0 + diagonal * sin_theta)
        y2 = int(y0 - diagonal * cos_theta)
        # plot line
        fig[1, 1].add_line(mlines.Line2D([x1, x2], [y1, y2]))
    plt.show()


def circle_detection(img, threshold=0.5, min_count=0, min_r=30, max_r=200, min_dis=50, blur=False):
    print("Preprocessing image...")
    edge_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.bilateralFilter(edge_image, 90, 75, 150)
    if blur:
        edge_image = cv2.medianBlur(edge_image, 9)
        _threshold, edge_image = cv2.threshold(edge_image, 128, 192, cv2.THRESH_OTSU)
        edge_image = np.uint8((edge_image > _threshold) * 255)
        edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=1)
        edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    if blur:
        edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Quantify parameter space
    height, width = edge_image.shape[:2]
    grad_x = cv2.Scharr(edge_image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(edge_image, cv2.CV_32F, 0, 1)
    _, angle_deg = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle_deg = (270 - angle_deg % 180) % 180
    angle = np.deg2rad(angle_deg)

    # Filling
    # normalization
    edge_points = np.argwhere(edge_image != 0)
    # b = r * cos(theta) + y
    radius = np.append(np.arange(-max_r, min_r), np.arange(min_r, max_r))
    b_values = np.matmul(np.append(radius.reshape((radius.shape[0], 1)), np.ones((radius.shape[0], 1)), axis=1),
                         np.vstack((np.cos(angle[edge_points.T[0], edge_points.T[1]]).reshape(1, edge_points.shape[0]),
                                    edge_points.T[0].reshape(edge_points.shape[0])))).transpose()
    a_values = np.matmul(np.append(radius.reshape((radius.shape[0], 1)), np.ones((radius.shape[0], 1)), axis=1),
                         np.vstack((np.sin(angle[edge_points.T[0], edge_points.T[1]]).reshape(1, edge_points.shape[0]),
                                    edge_points.T[1].reshape(edge_points.shape[0])))).transpose()
    # filling the accumulator with np.histogram2d()
    accumulator, _, _ = np.histogram2d(a_values.reshape(-1), b_values.reshape(-1), bins=[np.arange(0, width), np.arange(0, height)])
    auto_threshold = np.max([min_count, np.max(accumulator) * threshold])

    # Results
    # pick up parameters which counts are large enough
    print('Searching for centers...')
    voted_centers = np.argwhere(accumulator > auto_threshold)
    local_centers = np.empty((0, 3), int)
    for a, b in tqdm(voted_centers):
        _accumulator = np.zeros(max_r)
        x0 = max(0, a - 1)
        x1 = min(width, a + 1)
        y0 = max(0, b - 1)
        y1 = min(height, b + 1)
        if accumulator[a, b] == np.max(accumulator[x0:x1, y0:y1]):
            local_centers = np.append(local_centers, [[a, b, accumulator[a, b]]], axis=0)
    local_centers = local_centers[np.lexsort(np.transpose(local_centers))][::-1]

    print('Removing centers too close...')
    centers = np.array([local_centers[0, :-1]])
    for a, b, _ in tqdm(local_centers[1:]):
        distances = np.sqrt(np.sum(np.square(centers - (a, b)), axis=1))
        if np.min(distances) > min_r:
            centers = np.append(centers, [[a, b]], axis=0)

    print('Estimating radius...')
    real_centers = []
    for a, b in tqdm(centers):
        distances = np.sqrt(np.sum(np.square(edge_points[:, [1, 0]] - (a, b)), axis=1)).astype(int)
        radius = np.argmax(np.bincount(distances))
        if min_r < radius < max_r:
            real_centers.append((a, b, radius))
    real_centers = np.array(real_centers)

    print('Removing overlaps...')
    try:
        if blur:
            real_centers = real_centers[np.lexsort(np.transpose(real_centers))][::-1]
        final_centers = np.array([real_centers[0]])
        for a, b, r in tqdm(real_centers[1:]):
            distances = np.sqrt(np.sum(np.square(final_centers[:, [0, 1]] - (a, b)), axis=1))
            if np.min(distances) > min_dis:
                final_centers = np.append(final_centers, [[a, b, r]], axis=0)
    except IndexError:
        print('[!] No circles in this image!')
        final_centers = real_centers

    # Visualization
    # 4 subplots
    print("Plotting...")
    _, fig = plt.subplots(2, 2)
    fig[0, 0].imshow(img)
    fig[0, 0].set_title("Original Image")
    fig[0, 0].axis('off')
    fig[0, 1].imshow(edge_image, cmap="gray")
    fig[0, 1].set_title("Edge Image")
    fig[0, 1].axis('off')
    # fig[1, 0].set_facecolor((0, 0, 0))
    # plot parameters
    fig[1, 0].set_title("Hough Space (a, b)")
    fig[1, 0].imshow(accumulator.transpose(), cmap='hot', interpolation='nearest')
    # fig[1, 0].invert_yaxis()
    fig[1, 1].imshow(img)
    fig[1, 1].set_title("Detected Circles")
    fig[1, 1].axis('off')
    # plot circles
    for a, b, r in tqdm(final_centers):
        fig[1, 1].add_patch(Circle((a, b), r, fill=False, color='green'))
        fig[1, 1].plot([a], [b], marker='o', color='yellow')
        fig[1, 0].plot([a], [b], marker='o', color='yellow')
    plt.show()


if __name__ == "__main__":
    debug = False
    # lines
    line_detection(cv2.imread(f"assets/sample-1.jpg"), threshold=0.60)
    line_detection(cv2.imread(f"assets/sample-2.jpg"), num_rhos=720, num_thetas=720, threshold=0.30)
    line_detection(cv2.imread(f"assets/sample-3.jpg"), min_count=200)

    # circles
    circle_detection(cv2.imread(f"assets/sample-8.jpg"), threshold=0.47, min_r=36, max_r=50)
    circle_detection(cv2.imread(f"assets/sample-1.jpg"), threshold=0.20, min_r=25, max_r=140, min_dis=100, blur=True)
    circle_detection(cv2.imread(f"assets/sample-2.jpg"), threshold=0.90)
    circle_detection(cv2.imread(f"assets/sample-3.jpg"), threshold=0.10, min_r=2, max_r=80)
