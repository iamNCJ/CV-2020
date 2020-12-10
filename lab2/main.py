import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Circle


def line_detection(img, num_rhos=360, num_thetas=360, threshold=0.5, min_count=0):
    def _debug():
        if debug:
            cv2.imshow('res', edge_image)
            cv2.waitKey(-1)
    print("Preprocessing image...")
    edge_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _debug()
    edge_image = cv2.bilateralFilter(edge_image, 90, 75, 150)
    _debug()
    edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    _debug()
    edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    _debug()
    edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    _debug()
    edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    _debug()
    edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    _debug()

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


def circle_detection(img, threshold=0.5, min_count=0, min_r=10, max_r=100):
    def _debug():
        if debug:
            cv2.imshow('res', edge_image)
            cv2.waitKey(-1)
    print("Preprocessing image...")
    edge_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _debug()
    # edge_image = cv2.bilateralFilter(edge_image, 90, 75, 150)
    # _debug()
    # # edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    # # _debug()
    # edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    # _debug()
    # edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    _debug()
    edge_image = cv2.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
    _debug()
    # edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    # edge_image = cv2.erode(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    # _debug()

    # Quantify parameter space
    height, width = edge_image.shape[:2]
    grad_x = cv2.Scharr(edge_image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(edge_image, cv2.CV_32F, 0, 1)
    _, angle_deg = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle_deg = (270 - angle_deg % 180) % 180
    angle = np.deg2rad(angle_deg)
    # (a, b): circle center
    _as = np.arange(0, width)
    _bs = np.arange(0, height)

    # Filling
    # normalization
    edge_points = np.argwhere(edge_image != 0)
    # angles = [angle[tuple(x)] for x in edge_points]
    tans = np.tan(angle[edge_points.T[0], edge_points.T[1]])
    # b = a * tan(theta) - x * tan(theta) + y
    accumulator = np.zeros((width, height))
    for (y, x), tan in tqdm(zip(edge_points, tans)):
        # print(x, y)
        # bs = _as * tan - x * tan + y
        # bs = bs.astype(int)
        # b_ok = np.argwhere(0 <= bs < height)
        for a in _as:
            b = a * tan - x * tan + y
            if 0 <= b < height:
                accumulator[a, int(b)] += 1
        # accumulator[]
        # accumulator += 1
    # b_values = np.matmul(np.append(_as.reshape((_as.shape[0], 1)), np.ones((_as.shape[0], 1)), axis=1),
    #                      np.vstack((tans.reshape(1, tans.shape[0]), (-edge_points.T[0] * tans + edge_points.T[1]).reshape(tans.shape[0])))).transpose()
    #
    # b_values = b_values.astype(int)
    # # b_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    # # filling the accumulator with np.histogram2d()
    # accumulator, _, _ = np.histogram2d(np.tile(_as, b_values.shape[0]), b_values.reshape(-1), bins=[_as, _bs])
    auto_threshold = np.max([min_count, np.max(accumulator) * threshold])

    # Results
    # pick up parameters which counts are large enough
    centers = np.argwhere(accumulator > auto_threshold)
    real_centers = []
    for a, b in centers:
        accumulator = np.zeros(max_r)
        for y, x in edge_points:
            distance = int(np.sqrt(np.square(x - a) + np.square(y - b)))
            if min_r <= distance < max_r:
                accumulator[distance] += 1

    # a, b = centers[:, 0], centers[:, 1]
    # a, b = _as[a_idx], _bs[b_idx]

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
    # fig[1, 1].imshow(img)
    fig[1, 1].imshow(angle_deg, cmap='hot', interpolation='nearest')
    fig[1, 1].set_title("Detected Circles")
    fig[1, 1].axis('off')
    # plot parameters
    fig[1, 0].imshow(accumulator.transpose(), cmap='hot', interpolation='nearest')
    # for _b in tqdm(b_values):
    #     fig[1, 0].plot(_as, _b, color="white", alpha=0.01)
    # plot selected parameters
    fig[0, 0].plot(real_centers[:, 0], real_centers[:, 1], marker='o', color='yellow')
    for a, b, r in real_centers:
        fig[0, 0].add_patch(Circle((a, b), r))
    # plot lines
    # fig[1, 1].plot([a], [b], marker='o', color='yellow')
    # for a, b in zip(range(width), range(height)):
    #     cv2.drawMarker(img, (a, b), (int(angle[a, b] / 6.29 * 255), 0, 0))
    # fig[1, 1].imshow(img)
    # for _a, _b in zip(a, b):
    #     cos_theta = np.cos(np.deg2rad(_theta))
    #     sin_theta = np.sin(np.deg2rad(_theta))
    #     # calculate [x1, y1] and [x2, y2] from parametric functions
    #     x0 = (cos_theta * _rho) + width / 2
    #     y0 = (sin_theta * _rho) + height / 2
    #     x1 = int(x0 - diagonal * sin_theta)
    #     y1 = int(y0 + diagonal * cos_theta)
    #     x2 = int(x0 + diagonal * sin_theta)
    #     y2 = int(y0 - diagonal * cos_theta)
    #     # plot line
        fig[1, 1].add_line(mlines.Line2D([x1, x2], [y1, y2]))
    plt.show()


if __name__ == "__main__":
    debug = False
    # line_detection(cv2.imread(f"assets/sample-1.jpg"), threshold=0.60)
    # line_detection(cv2.imread(f"assets/sample-2.jpg"), num_rhos=720, num_thetas=720, threshold=0.30)
    # line_detection(cv2.imread(f"assets/sample-3.jpg"), min_count=200)
    circle_detection(cv2.imread(f"assets/sample-6.jpg"), threshold=0.60)
