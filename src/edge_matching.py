
from ast import Match
import os

from numpy import shape
from puzzle_types import *


def find_matches_closest(a, b, transform):
    matches = []
    for point_b in b:
        point_b_transformed = transform(point_b.squeeze())  # Ensure (2,)
        closest_point = min(a, key=lambda p: np.linalg.norm(point_b_transformed - p.squeeze()))
        matches.append((closest_point.squeeze(), point_b.squeeze()))
    return matches


def find_transformation_LSQ(matches):

    if not matches:
        return Transform()

    a_points = np.array([m[0] for m in matches])
    b_points = np.array([m[1] for m in matches])
    
    a_mean = np.mean(a_points, axis=0)
    b_mean = np.mean(b_points, axis=0)
    
    s_xx = s_yy = s_xy = s_yx = 0.0
    for a, b in matches:
        a = np.squeeze(a)
        b = np.squeeze(b)
        
        if a.shape != (2,) or b.shape != (2,):  # Ensure correct shape
            raise ValueError(f"Unexpected point shape: a={a.shape}, b={b.shape}")
        
        dx_a, dy_a = a - a_mean
        dx_b, dy_b = b - b_mean
        
        s_xx += dx_b * dx_a
        s_yy += dy_b * dy_a
        s_xy += dx_b * dy_a
        s_yx += dy_b * dx_a
    
    w = np.arctan2(s_xy - s_yx, s_xx + s_yy)
    rotation_matrix = np.array([
        [np.cos(w), -np.sin(w)], 
        [np.sin(w), np.cos(w)]
    ])
    
    t = a_mean - np.dot(rotation_matrix, b_mean)
    
    return Transform(t, w)


def compare_edges(a, b):
    if a.type == b.type or a.type == 'Flat' or b.type == 'Flat':
        return float('inf')
    
    transform = find_transformation_LSQ([
        (np.squeeze(a.points[0]), np.squeeze(b.points[-1])),
        (np.squeeze(a.points[-1]), np.squeeze(b.points[0]))
    ])
    
    for i in range(10):
        matches = find_matches_closest(a.points, b.points, transform)
        new_transform = find_transformation_LSQ(matches)
        if new_transform == transform and i > 0:
            break
        transform = new_transform
        
        if i == 0:
            sum_sq_dist = sum(np.linalg.norm(m[0] - transform(m[1]))**2 for m in matches)
            if sum_sq_dist > 10_000:
                return sum_sq_dist
    
    sum_sq_dist = sum(np.linalg.norm(m[0] - transform(m[1]))**2 for m in matches)
    print(sum_sq_dist)
    return sum_sq_dist