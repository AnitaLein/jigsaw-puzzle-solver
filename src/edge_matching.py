
from ast import Match
import os

from numpy import shape
from puzzle_types import *

"""
def find_matches_closest(a, b, transform):
    matches = []
    for aPoint in a:
        for bPoint in b:
            transformed = transform(aPoint)
            distance = np.linalg.norm(bPoint - transformed)
            matches.append(Match(aPoint, bPoint, distance))"""

def find_matches_closest(a, b, transform):
    matches = []
    for point_b in b:
        point_b_transformed = transform(point_b)
        
        # Calculate the distances and find the closest point in 'a'
        distances = [np.linalg.norm(np.array(point_b_transformed) - np.array(point_a)) for point_a in a]
        closest_index = np.argmin(distances)
        matches.append((a[closest_index], point_b))
    
    return matches


def find_transformation_lsq(matches):
    transform = Transform()
    
    if not matches or len(matches) < 2:
        raise ValueError("Not enough matches for transformation")
    
    a_points = np.array([np.asarray(match[0], dtype=np.float64).flatten() for match in matches])
    b_points = np.array([np.asarray(match[1], dtype=np.float64).flatten() for match in matches])
    
    if a_points.shape[1] != 2 or b_points.shape[1] != 2:
        raise ValueError("Invalid match dimensions: Expected 2D points")
    
    a_mean = np.mean(a_points, axis=0)
    b_mean = np.mean(b_points, axis=0)
    
    s_xx = s_yy = s_xy = s_yx = 0.0
    
    for a, b in matches:
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        s_xx += (b[0] - b_mean[0]) * (a[0] - a_mean[0])
        s_yy += (b[1] - b_mean[1]) * (a[1] - a_mean[1])
        s_xy += (b[0] - b_mean[0]) * (a[1] - a_mean[1])
        s_yx += (b[1] - b_mean[1]) * (a[0] - a_mean[0])
    
    transform.w = math.atan2(s_xy - s_yx, s_xx + s_yy)
    transform.t = a_mean - np.array([
        b_mean[0] * math.cos(transform.w) - b_mean[1] * math.sin(transform.w),
        b_mean[0] * math.sin(transform.w) + b_mean[1] * math.cos(transform.w)
    ])
    
    return transform

def compare_edges(a, b):
    global n
    n = getattr(compare_edges, 'n', 0)
    
    if a.type == b.type or a.type == "Flat" or b.type == "Flat":
        return float('inf')
    
    transform = find_transformation_lsq([(a.points[0], b.points[-1]), (a.points[-1], b.points[0])])
    
    for i in range(10):
        matches = find_matches_closest(a.points, b.points, transform)
        
        if not matches:
            return float('inf')
        
        new_transform = find_transformation_lsq(matches)
        
        if new_transform == transform and i > 0:
            break
        
        transform = new_transform
        
        if i == 0:
            sum_sq = sum(np.linalg.norm(np.asarray(m[0]) - transform(np.asarray(m[1])))**2 for m in matches)
            if sum_sq > 10_000:
                return sum_sq
    
    sum_sq = sum(np.linalg.norm(np.asarray(m[0]) - transform(np.asarray(m[1])))**2 for m in matches)
    
    if 2500 < sum_sq < 3000:
        image = np.zeros((600, 600, 3), dtype=np.uint8)
        for i in range(len(a.points) - 1):
            cv2.line(image, tuple(a.points[i]), tuple(a.points[i + 1]), (0, 255, 0), 1)
        for i in range(len(b.points) - 1):
            cv2.line(image, tuple(transform(b.points[i])), tuple(transform(b.points[i + 1])), (0, 0, 255), 1)
        cv2.imwrite(f"eda2/match_{n}.png", image)
        
        image = np.zeros((600, 600, 3), dtype=np.uint8)
        for i in range(len(a.points) - 1):
            cv2.line(image, tuple(a.points[i]), tuple(a.points[i + 1]), (0, 255, 0), 1)
        for i in range(len(b.points) - 1):
            cv2.line(image, tuple(transform(b.points[i]) + np.array([0, 50])), tuple(transform(b.points[i + 1]) + np.array([0, 50])), (0, 0, 255), 1)
        cv2.imwrite(f"eda2/match_offset_{n}.png", image)
        
        n += 1
    
    return sum_sq