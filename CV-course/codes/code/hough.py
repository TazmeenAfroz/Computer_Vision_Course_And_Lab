import numpy as np
import cv2

def hough_transform(edges, threshold=100):
    """
    Step 2: Implement Hough Transform from scratch.
    Input:
        edges     -> binary edge image (from Canny)
        threshold -> minimum votes in accumulator to consider a line
    Output:
        accumulator -> 2D Hough space array
        rhos        -> array of rho values
        thetas      -> array of theta values
        lines       -> list of (rho, theta) pairs that passed threshold
    """

    # --- Step 1: Setup rho and theta ranges ---
    height, width = edges.shape
    
    # Maximum possible rho is the image diagonal length
    max_rho = int(np.ceil(np.sqrt(height**2 + width**2)))

    # Rho values: from -max_rho to +max_rho
    rhos = np.arange(-max_rho, max_rho + 1, 1)

    # Theta values: angles from 0 to 180 degrees
    thetas = np.deg2rad(np.arange(0, 180, 1))
    
    # Accumulator matrix of size (#rhos × #thetas)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # --- Step 2: Get edge (x,y) coordinates ---
    y_coords, x_coords = np.nonzero(edges)
    
    # --- Step 3: Voting in Hough Space ---
    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]
        for t_idx in range(len(thetas)):
            theta = thetas[t_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            rho_idx = rho + max_rho
            accumulator[rho_idx, t_idx] += 1
    
    # --- Step 4: Extract lines above threshold ---
    lines = []
    for r_idx in range(len(rhos)):
        for t_idx in range(len(thetas)):
            if accumulator[r_idx, t_idx] > threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                lines.append((rho, theta))
    
    return accumulator, rhos, thetas, lines


def draw_lines(img, lines):
    """
    Step 3: Draw detected lines on a copy of the image.
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)

        # --- Handle vertical and horizontal lines safely ---
        if np.abs(b) < 1e-6:  # nearly vertical line
            x = int(rho / a)
            cv2.line(img_color, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)
        elif np.abs(a) < 1e-6:  # nearly horizontal line
            y = int(rho / b)
            cv2.line(img_color, (0, y), (img.shape[1], y), (0, 0, 255), 2)
        else:
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return img_color

