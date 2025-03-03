

#from arrange_pieces import *
from collections import defaultdict
from csv_saving import *
from puzzle_types import EdgeType
from puzzle_types import *
import csv



def setFirstPiece(grid, puzzle_pieces):
    appended = []
    center = len(grid[0]) // 2
    pos_x, pos_y = center, center
    grid[pos_y][pos_x] = (puzzle_pieces[0] , 0)
    appended.append((puzzle_pieces[0], pos_y, pos_x))
    return grid, appended

def iterateOverAppended(appended, grid, puzzle_pieces, similarity_matrix):
    grid, appended = setFirstPiece(grid, puzzle_pieces)
    for i in range(len(puzzle_pieces)):
        grid, appended = solvePuzzle(similarity_matrix, puzzle_pieces, grid, appended, i)
        
    return grid

def solvePuzzle(similarity_matrix, puzzle_pieces, grid, appended, iteration = 0):
    for i in range(4):
        print('iteration', iteration)
        print('appended', appended)
        pos_y, pos_x = appended[iteration][1], appended[iteration][2] 
        rotation = grid[pos_y][pos_x][1]

        piece_i = puzzle_pieces.index(appended[iteration][0])

        row = similarity_matrix[i + (piece_i * 4)]
        matches = [(row[j], j) for j in range(len(row)) if row[j] != float('inf')]
        if len(matches) == 0:
            continue
        matches = sorted(matches, key=lambda tup: tup[0])
        print('matches', matches, i)
        for match in matches:
            matching_piece = puzzle_pieces[match[1] // 4]
            
            if all(matching_piece != x[0] for x in appended):
                print('matching piece', matching_piece)
                matching_edge = match[1] % 4
                matching_rotation = piece_rotation((i + rotation), matching_edge)
                if (i + rotation) % 4 == 0: 
                    pos_y -= 1
                elif (i + rotation) % 4 == 1:
                    pos_x += 1
                elif (i + rotation) % 4 == 2:
                    pos_y += 1
                elif (i + rotation)% 4 == 3:
                    pos_x -= 1

                if grid[pos_y][pos_x] is None:
                    #grid[pos_y][pos_x] = (matching_piece, matching_rotation)
                    #appended.append((matching_piece, pos_y, pos_x))
                    grid, appended = check_surr_pieces(grid, puzzle_pieces, similarity_matrix, pos_y, pos_x, appended, matching_piece, matching_rotation)
                    break
                else:    
                    #grid, appended = check_other_edges(puzzle_pieces, similarity_matrix, grid, pos_y, pos_x, appended)
                    print('fixed appended', appended)
                    break
            else:
                print('Piece already appended')
                continue
    return grid, appended


def check_surr_pieces(grid, puzzle_pieces, similarity_matrix, pos_y, pos_x, appended, matching_piece, matching_rotation):
    surr = []
    # (piece, edfe before rotation, edge after rotation)
    if grid[pos_y + 1][pos_x] is not None:
        surr.append((grid[pos_y + 1][pos_x][0], (0 - grid[pos_y + 1][pos_x][1]) % 4, 0))
    if grid[pos_y - 1][pos_x] is not None:
        surr.append((grid[pos_y - 1][pos_x][0], (2 - grid[pos_y - 1][pos_x][1]) % 4, 2))
    if grid[pos_y][pos_x + 1] is not None:
        surr.append((grid[pos_y][pos_x + 1][0], (3 - grid[pos_y][pos_x + 1][1]) % 4, 3))
    if grid[pos_y][pos_x - 1] is not None:
        surr.append((grid[pos_y][pos_x - 1][0], (1 - grid[pos_y][pos_x - 1][1]) % 4, 1))

    print('surr', surr)
    if len(surr) == 1:
        if matching_piece not in appended:
            grid[pos_y][pos_x] = (matching_piece, matching_rotation)
            appended.append((matching_piece, pos_y, pos_x))
            return grid, appended
        else:
            return grid, appended
        
    
    all_matches = []
    for surr_el in surr:
        piece_i = puzzle_pieces.index(surr_el[0])
        edge = surr_el[1]
        row = similarity_matrix[edge + (piece_i*4)]
        matches = [(row[j], j) for j in range(len(row)) if row[j] != float('inf')]
        print('matches', matches)
        if len(matches) == 0:
            continue
        matches = sorted(matches, key=lambda tup: tup[0])
        all_matches.append((matches, surr_el))
    print('all matches', all_matches)
    piece_i, rotation = find_common_elements(all_matches, puzzle_pieces, appended, matching_piece, matching_rotation)
    grid[pos_y][pos_x] = (puzzle_pieces[piece_i], rotation)
    appended.append((puzzle_pieces[piece_i], pos_y, pos_x))
    return grid, appended
    # check if there is a common element in all matches



def find_common_elements(lists, puzzle_pieces, appended, matching_piece, matching_rotation):
    # Convert each list into a new format (best match, piece index, edge index, neighbor)
    #print('lists', lists)
    new_lists = [
        #((best match, macthing piece index, matching edge index), 
        # (surr piece, surr edge before rotation, surr edge after rotation), list index)
        ([(t[0], t[1] // 4, t[1] % 4) for t in lst[0]], lst[1], lists.index(lst))
        for lst in lists
    ]
    #print('new lists', new_lists)
    
    # Get the piece numbers that are in all the best matches
    filtered = [t[0]for t in new_lists]
    filtered = [list(map(lambda x: x[1], lst)) for lst in filtered]
    common_pieces = list(set.intersection(*map(set, filtered)))  # Find common pieces across all lists

    if not common_pieces:
        index = puzzle_pieces.index(matching_piece)
        return index, matching_rotation
    print('common pieces', common_pieces)
    # Map the best matches for each common piece
    filtered_best_matches = {piece: [] for piece in common_pieces}
    
    #print('new lists', new_lists)  
    for lst in new_lists:
        for el in lst[0]:
            if el[1] in common_pieces:
                filtered_best_matches[el[1]].append((el, lst[1], lst[2]))

    print('filtered best matches', filtered_best_matches)
    results = defaultdict(set)
    for key, elements in filtered_best_matches.items():
        # Split elements into two lists based on index
        list1 = [e for e in elements if e[2] == 0]
        list2 = [e for e in elements if e[2] == 1]
        print('list1', list1)
        print('list2', list2)
        
        if not list1 or not list2:
            continue  # Skip if one of the lists is empty
        
        # Extract rotations
        for i in range(len(list1)):
            for j in range(len(list2)):
                edge1 = next(iter({e[1][2] for e in list1}))  # Convert set to single value
                edge2 = next(iter({e[1][2] for e in list2}))
                print('edge1', edge1)
                print('edge2', edge2)
                rotation1 = piece_rotation(list1[i][0][2], edge1)
                rotation2 = piece_rotation(list2[j][0][2], edge2)
                print('rotation1', rotation1)
                print('rotation2', rotation2)
                if rotation1 == rotation2:
                    sum_score = list1[i][0][0] + list2[j][0][0]
                    results[key].add((sum_score, rotation1))
                else:
                    print('skipped')
                    continue


    if results:
        flattened_list = [(x[0],x[1], key) for key, value in results.items() for x in value]
        sorted_flattened_list = sorted(flattened_list, key=lambda x: x[0])
        for el in sorted_flattened_list:
            if puzzle_pieces[el[2]] not in appended:
                return el[2], el[1]
            else:
                continue

def rotate_piece(piece, rotation):
    new_piece = PuzzlePiece(piece.number, piece.type, edges=[])
    for i in range(4):
        new_piece.edges.append(piece.edges[(i + rotation) % 4])

    return new_piece

def corner_rotation(a):
    if a == [0, 1]:
        return 3
    elif a == [1, 2]:
        return 2
    elif a == [2, 3]:
        return 1
    elif a == [3,0]:
        return 0

#how many times needs to be rotated
def piece_rotation(a, b):
    return (a - b + 2) % 4


def find_best_match(puzzle_piece, edge_index, similarity_matrix):
    best_match_i = None
    min_dist = float('inf')
    i = int(puzzle_piece.number)

    row = similarity_matrix[i+edge_index]
    for j in range(len(row)):
        if row[j] < min_dist:
            min_dist = row[j]
            best_match_i = j
    return (min_dist, best_match_i)

def get_matches(puzzle_piece, edge_index, similarity_matrix):
    i = int(puzzle_piece.number)
    row = similarity_matrix[i + edge_index]  # Get the corresponding row

    # Collect all non-infinity values along with their original indices
    matches = [(row[j], j) for j in range(len(row)) if row[j] != float('inf')]

    # Sort matches by similarity score (lower is better)
    matches.sort()

    return matches
