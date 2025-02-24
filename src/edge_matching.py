
from ast import Match
import os

from numpy import shape
from puzzle_types import *
from scipy.spatial import cKDTree


def find_matches_closest(a, b, transform):
    b_transformed = np.array([transform(p.squeeze()) for p in b])
    a_points = np.array([p.squeeze() for p in a])

    tree = cKDTree(a_points)
    _, indices = tree.query(b_transformed)

    matches = [(a_points[idx], b.squeeze()) for idx, b in zip(indices, b)]
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
    return sum_sq_dist

def edge_matching(puzzle_pieces):
   
    for i in range(0, len(puzzle_pieces)):
        for x in range(0, 4):
            match = Match(0,None,0,0)
            best_sqr = float('inf')
            if puzzle_pieces[i].edges[x].type == EdgeType.Gerade:
                continue
            for j in range(i + 1, len(puzzle_pieces)):
                for y in range(0, 4):
                    if  puzzle_pieces[j].edges[y].type == EdgeType.Gerade or puzzle_pieces[i].edges[x].type == puzzle_pieces[j].edges[y].type:
                        continue
                    sum_sqr= compare_edges(puzzle_pieces[i].edges[x], puzzle_pieces[j].edges[y])
                    if sum_sqr < best_sqr:
                        best_sqr = sum_sqr
                        best_piece_index = j
                        best_edge_index = y
            
            match.matching_piece = best_piece_index
            match.matching_edge = puzzle_pieces[best_piece_index].edges[best_edge_index]
            match.originalEdge = x
            match.matchingEdge = best_edge_index
            puzzle_pieces[i].matches.append(match)
        
    return puzzle_pieces

def compute_similarity_matrix(puzzle_pieces):
    similarity_matrix = []

    for i in range(len(puzzle_pieces)):
        for x in range(4):
            similarities = []

            for j in range(len(puzzle_pieces)):
                for y in range(4):
                    # Check edge continuity
                    if ((puzzle_pieces[i].edges[(x + 1) % 4].type == EdgeType.Gerade) != 
                        (puzzle_pieces[j].edges[(y + 3) % 4].type == EdgeType.Gerade) or
                        (puzzle_pieces[i].edges[(x + 3) % 4].type == EdgeType.Gerade) != 
                        (puzzle_pieces[j].edges[(y + 1) % 4].type == EdgeType.Gerade)):
                        similarity = float("inf")
                    else:
                        similarity = compare_edges(puzzle_pieces[i].edges[x], puzzle_pieces[j].edges[y])

                    similarities.append(similarity)
            

            similarity_matrix.append(similarities)
        
        print(".", end="", flush=True)

    return np.array(similarity_matrix)