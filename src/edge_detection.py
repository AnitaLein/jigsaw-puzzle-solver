import glob
import os
import cv2
import numpy as np

# Define thresholds
LENGTH_THRESHOLD = 50  # Minimum length of an edge (in pixels)
CURVATURE_THRESHOLD = 0.05  # Curvature threshold for flat vs curved edges
MSE_THRESHOLD = 10  # Mean Squared Error threshold for flatness

# Function to calculate curvature
def calculate_curvature(edge_points):
    """
    Calculate curvature as the inverse of the radius of the circle
    passing through three consecutive points.
    """
    if len(edge_points) < 3:  # Not enough points to calculate curvature
        return 0

    curvatures = []
    for i in range(len(edge_points) - 2):
        p1 = edge_points[i][0]
        p2 = edge_points[i + 1][0]
        p3 = edge_points[i + 2][0]
        
        # Use determinant to calculate radius
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        if a == 0 or b == 0 or c == 0:
            continue
        
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        if area == 0:
            continue
        
        radius = (a * b * c) / (4 * area)
        curvature = 1 / radius
        curvatures.append(curvature)
    
    # Return average curvature for the edge
    return np.mean(curvatures) if curvatures else 0

# Function to classify edges
def classify_edge(edge_points):
    # Calculate edge length
    edge_length = cv2.arcLength(edge_points, closed=False)
    if edge_length < LENGTH_THRESHOLD:
        return "Short edge (noise)"

    # Fit a straight line and calculate MSE
    [vx, vy, x0, y0] = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)
    distances = []
    for pt in edge_points:
        x, y = pt[0]
        distance = abs((vy * (x - x0) - vx * (y - y0)) / np.sqrt(vx**2 + vy**2))
        distances.append(distance)
    mse = np.mean(np.square(distances))

    # Calculate curvature
    curvature = calculate_curvature(edge_points)

    # Apply thresholds
    if mse < MSE_THRESHOLD:
        return "Flat edge"
    elif curvature > CURVATURE_THRESHOLD:
        return "Tab"
    else:
        return "Slot"

img = cv2.imread('../data/11b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

# filter  
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

# canny edge detection
edges = cv2.Canny(img, 100, 200)
edges_bilateralblur = cv2.Canny(bilateral_blur, 100, 200)

# find contours
contours, _ = cv2.findContours(edges_bilateralblur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# filter contours by area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

# save edge_images in this folder
output_folder = 'output_edges_contours'

# delete existing data in the folder
files = glob.glob(os.path.join(output_folder, '*.png'))
for f in files:
    os.remove(f)

# extract and save edges
for i, cnt in enumerate(filtered_contours):
    # create an empty image and draw the contour
    edge_image = np.zeros_like(edges_bilateralblur)
    cv2.drawContours(edge_image, [cnt], -1, 255, 1)
    
    # save and visualize the edge image
    output_filename = os.path.join(output_folder, f'output_edges_contour_{i}.png')
    cv2.imwrite(output_filename, edge_image)
    #cv.imshow(f'Contour {i}', edge_image)


input_folder = 'output_edges_contours'
output_folder = 'output_corners'
input_images = glob.glob(os.path.join(input_folder, '*.png'))

# Process each contour
for contour in contours:
    # Approximate the contour for smoother edges
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Visualize the contour
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
    # Classify the edge
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        edge_points = np.array([pt1, pt2]).reshape(-1, 1, 2)

        # Classify each edge
        edge_class = classify_edge(edge_points)

        # Annotate the edge on the image
        midpoint = (pt1 + pt2) // 2
        if(edge_class == "Flat edge"):
            cv2.putText(img, edge_class, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


# Show results
cv2.imshow("Puzzle Edge Classification", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
