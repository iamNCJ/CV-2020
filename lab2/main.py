import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def line_detection_vectorized(image, edge_image, num_rhos=360, num_thetas=360, t_count=530):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    
    _, fig = plt.subplots(2, 2)
    fig[0, 0].imshow(image)
    fig[0, 0].set_title("Original Image")
    fig[0, 0].axis('off')

    fig[0, 1].imshow(edge_image, cmap="gray")
    fig[0, 1].set_title("Edge Image")
    fig[0, 1].axis('off')

    fig[1, 0].set_facecolor((0, 0, 0))
    fig[1, 0].set_title("Hough Space")
    fig[1, 0].invert_yaxis()
    fig[1, 0].invert_xaxis()

    fig[1, 1].imshow(image)
    fig[1, 1].set_title("Detected Lines")
    fig[1, 1].axis('off')

    edge_points = np.argwhere(edge_image != 0)
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])

    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))

    accumulator, theta_vals, rho_vals = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )
    accumulator = np.transpose(accumulator)
    # print(np.max(accumulator))
    auto_threshold = np.max([t_count, np.max(accumulator) * 0.36])
    lines = np.argwhere(accumulator > auto_threshold)
    rho_idxs, theta_idxs = lines[:, 0], lines[:, 1]
    r, t = rhos[rho_idxs], thetas[theta_idxs]

    print("Plotting in hough space...")
    for ys in tqdm(rho_values):
        fig[1, 0].plot(thetas, ys, color="white", alpha=0.01)

    fig[1, 0].plot([t], [r], color="yellow", marker='o')

    for line in lines:
        y, x = line
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        fig[1, 0].plot([theta], [rho], marker='o', color="yellow")
        fig[1, 1].add_line(mlines.Line2D([x1, x2], [y1, y2]))

    plt.show()
    return accumulator, rhos, thetas


if __name__ == "__main__":
    def _debug():
        if debug:
            cv2.imshow('res', edge_image)
            cv2.waitKey(-1)
    debug = False
    for i in range(3):
        image = cv2.imread(f"assets/sample-{i + 1}.jpg")
        edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        line_detection_vectorized(image, edge_image)
