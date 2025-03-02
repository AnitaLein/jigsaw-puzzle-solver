

#from arrange_pieces import *
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
                    #check_surr_pieces(grid, puzzle_pieces, similarity_matrix, pos_y, pos_x, appended)
                    grid[pos_y][pos_x] = (matching_piece, matching_rotation)
                    appended.append((matching_piece, pos_y, pos_x))
                    break
                else:    
                    #grid, appended = check_other_edges(puzzle_pieces, similarity_matrix, grid, pos_y, pos_x, appended)
                    print('fixed appended', appended)
                    break
            else:
                print('Piece already appended')
                continue
    return grid, appended


def check_surr_pieces(grid, puzzle_pieces, similarity_matrix, pos_y, pos_x, appended):
    surr = []
    if grid[pos_y + 1][pos_x] is not None:
        surr.append((grid[pos_y + 1][pos_x][0], 0 - grid[pos_y + 1][pos_x][1] % 4))
    if grid[pos_y - 1][pos_x] is not None:
        surr.append((grid[pos_y - 1][pos_x][0], 2 - grid[pos_y - 1][pos_x][1] % 4))
    if grid[pos_y][pos_x + 1] is not None:
        surr.append((grid[pos_y][pos_x + 1][0], 3 - grid[pos_y][pos_x + 1][1] % 4))
    if grid[pos_y][pos_x - 1] is not None:
        surr.append((grid[pos_y][pos_x - 1][0], 1 - grid[pos_y][pos_x - 1][1] % 4))
    if len(surr) == 1:
        return
    print('surr', surr)
    all_matches = []
    for x in surr:
        piece_i = puzzle_pieces.index(x[0])
        edge = x[1]
        row = similarity_matrix[edge + (piece_i*4)]
        matches = [(row[j], j) for j in range(len(row)) if row[j] != float('inf')]
        print('matches', matches)
        if len(matches) == 0:
            continue
        matches = sorted(matches, key=lambda tup: tup[0])
        all_matches.append(matches)
    
    if find_common_elements(all_matches):
        for x in find_common_elements(all_matches):
            piece = puzzle_pieces[x[1] // 4]

    # check if there is a common element in all matches
    
    print('all matches', all_matches)


def find_common_elements(lists):
    print('lists', lists)
    if len(lists) < 2:
        return None
    
    new_lists = []
    for lst in lists:
        new_lists.append([(t[0],t[1] // 4, t[1] % 4) for t in lst])
    print('new lists', new_lists)
    
    second_elements_divided = [
    {t[1] for t in lst}  # Perform integer division on second elements
    for lst in new_lists
]   
    common_second_elements = set.intersection(*second_elements_divided)
    print('common second', common_second_elements)

    common_tuples = [[] for _ in new_lists]
    for x in common_second_elements:
        # find x in new_lists
        for lst in new_lists:
            for t in lst:
                if t[1] == x:
                    common_tuples[new_lists.index(lst)].append(t)
        print('common tuples', common_tuples)

    for i, list1 in enumerate(common_tuples):
        for j, tuple1 in enumerate(list1):
            # Extract second element from tuple in list1
            second_element1 = tuple1[1]
            
            # Iterate through all other lists (including itself) and compare
            for k, list2 in enumerate(common_tuples):
                for l, tuple2 in enumerate(list2):
                    # Extract second element from tuple in list2
                    second_element2 = tuple2[1]
                    
                    # Perform the comparison
                    if second_element1 == second_element2:
                        rotation_element1 = piece_rotation(tuple1[2], second_element1)
                        rotation_element2 = piece_rotation(tuple2[2], second_element2)
                        if rotation_element1 == rotation_element2:
                            print(f"Right elemenrs: {second_element1} from list {i} tuple {j} == {second_element2} from list {k} tuple {l}")
                            break
                            #return tuple1, tuple2
                        else:
                            continue
                    else:
                        continue
        #return common_tuples

    
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
