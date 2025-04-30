import numpy as np
import matplotlib.pyplot as plt
import heapq
import copy
from itertools import permutations

# Vectorized utility functions
def can_place(board, x, y, z, h, w, d):
    # Convert board to numpy array if it's not already
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    H, W, D = board_np.shape
    if x + h > H or y + w > W or z + d > D:
        return False
    return np.all(board_np[x:x+h, y:y+w, z:z+d] == 0)

def place_block(board, x, y, z, h, w, d, block_id):
    # Convert board to numpy array if it's not already
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    board_np[x:x+h, y:y+w, z:z+d] = block_id
    # If board is not numpy array, update the original list
    if not isinstance(board, np.ndarray):
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    board[x+i][y+j][z+k] = block_id

def is_grounded(board, x, y, z, h, w, d):
    if x == 0:
        return True
    # Convert board to numpy array if it's not already
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    bottom_slice = board_np[x, y:y+w, z:z+d]
    under_slice = board_np[x-1, y:y+w, z:z+d]
    return np.all((bottom_slice == 0) | (under_slice != 0))

def compute_area(board):
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    return int(np.sum(board_np != 0))

def current_max_height(board):
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    H = board_np.shape[0]
    for x in range(H-1, -1, -1):
        if np.any(board_np[x] != 0):
            return x + 1
    return 0

def bottom_fill_score(board):
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    H, W, D = board_np.shape
    score = 0
    for x in range(H):
        filled = np.sum(board_np[x] != 0)
        score += filled * (H - x - 1)
    return score

def board_to_tuple(board):
    # Using the array's tobytes method for faster hashing
    if isinstance(board, np.ndarray):
        return board.tobytes()
    return tuple(tuple(tuple(layer) for layer in row) for row in board)

def compute_score(board):
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    area = compute_area(board_np)
    height = current_max_height(board_np)
    bottom_score = bottom_fill_score(board_np)

    H, W, D = board_np.shape
    floating_holes = 0
    # Vectorized hole detection
    for x in range(H):
        empty_cells = board_np[x] == 0
        if np.any(empty_cells):
            for xx in range(x+1, H):
                floating_holes += np.sum((board_np[x] == 0) & (board_np[xx] != 0))
                break

    return bottom_score + area - height# * 10 - floating_holes * 100 + placement_bonus

def find_lowest_grounded_x(board, y, z, h, w, d, H):
    for x in range(H - h + 1):
        if can_place(board, x, y, z, h, w, d) and is_grounded(board, x, y, z, h, w, d):
            return x
    return None

def tetris_beam_search(blocks, W=5, H=5, D=5, allow_rotations=False, beam_width=10, max_retries=3):
    initial_board = np.zeros((H, W, D), dtype=int)
    initial_state = (compute_score(initial_board), initial_board, tuple(blocks))

    for retry in range(max_retries):
        base_beam_width = beam_width * (2 ** retry)
        print(f"Beam search attempt {retry+1} with base beam width {base_beam_width}...")

        heap = [(initial_state[0], 0, initial_state[1], initial_state[2])]  # Add a unique ID as the second element
        best_score = -float('inf')
        best_board = None
        state_counter = 1  # Counter to generate unique IDs for states

        while heap:
            new_heap = []
            seen = set()

            for _, _, board, remaining in heap:
                score = compute_score(board)
                if score > best_score:
                    print("Found a solution with score:", score)
                    # visualize_board_voxels(board)
                    best_score = score
                    best_board = board.copy() if isinstance(board, np.ndarray) else copy.deepcopy(board)

                for idx in range(len(remaining)):
                    block = remaining[idx]
                    dims_list = [block]
                    if allow_rotations:
                        dims_list = list(set(permutations(block)))

                    for dims in dims_list:
                        h, w, d = dims
                        for y in range(W - w + 1):
                            for z in range(D - d + 1):
                                x = find_lowest_grounded_x(board, y, z, h, w, d, H)
                                if x is not None:
                                    if can_place(board, x, y, z, h, w, d) and is_grounded(board, x, y, z, h, w, d):
                                        # Use numpy's copy for faster duplication
                                        temp_board = board.copy() if isinstance(board, np.ndarray) else np.array(board)
                                        block_id = len(blocks) - len(remaining) + 1
                                        place_block(temp_board, x, y, z, h, w, d, block_id)
                                        # visualize_board_voxels(temp_board)
                                        new_score = compute_score(temp_board)
                                        new_remaining = remaining[:idx] + remaining[idx+1:]
                                        board_key = (board_to_tuple(temp_board), new_remaining)
                                        if board_key not in seen:
                                            seen.add(board_key)
                                            # Use the counter to ensure unique comparison for heap
                                            heapq.heappush(new_heap, (-new_score, state_counter, temp_board, new_remaining))
                                            state_counter += 1

            if not new_heap:
                break

            # Sort by score only, ignoring the counter
            sorted_heap = sorted(new_heap, key=lambda s: s[0])
            cutoff_score = sorted_heap[min(len(sorted_heap), base_beam_width)-1][0]
            heap = [s for s in sorted_heap if s[0] <= cutoff_score]  # tighter pruning with negative scores

        if best_board is not None:
            return best_board

    print("Warning: No valid complete solution found after retries!")
    return np.zeros((H, W, D), dtype=int)

def visualize_board_voxels(board):
    board_np = np.array(board) if not isinstance(board, np.ndarray) else board
    H, W, D = board_np.shape
    filled = board_np != 0
    colors = np.empty((H, W, D), dtype=object)

    for x in range(H):
        for y in range(W):
            for z in range(D):
                if board_np[x, y, z] != 0:
                    colors[x, y, z] = plt.cm.tab20(board_np[x, y, z] % 20)

    filled_plot = np.transpose(filled, (1, 2, 0))
    colors_plot = np.transpose(colors, (1, 2, 0))
    filled_plot = filled_plot[:, :, ::-1]
    colors_plot = colors_plot[:, :, ::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled_plot, facecolors=colors_plot, edgecolor='k')
    ax.set_xlabel('Width (Y)')
    ax.set_ylabel('Depth (Z)')
    ax.set_zlabel('Height (X)')
    # max_range = max(H, W, D)
    ax.set_xlim(0, D)
    ax.set_ylim(0, W)
    ax.set_zlim(H, 0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    def run_test_case(name, blocks, W, H, D, allow_rotations=True, beam_width=5):
        print(f"\n=== Test Case: {name} ===")
        board = tetris_beam_search(blocks, W=W, H=H, D=D, allow_rotations=allow_rotations, beam_width=beam_width)
        if board is not None:
            print(f"Best board found with score: {compute_score(board)}")
            print(f"Bottom fill score: {bottom_fill_score(board)}")
            print(f"Filled volume: {compute_area(board)}/{W*H*D}")
            print(f"Block Volume: {compute_area(board)}/{sum(h * w * d for h, w, d in blocks)}")
            print(f"Maximum height: {current_max_height(board)}/{H}")
            visualize_board_voxels(board)
        else:
            print("No valid board to visualize.")

    # test_case_small_perfect = [(2, 2, 2), (2, 3, 1), (1, 3, 2), (2, 3, 1)]
    # run_test_case("Small Perfect Fit", test_case_small_perfect, W=3, H=3, D=3)

    test_case_tall_thin = [(4, 1, 1), (1, 1, 4), (1, 4, 1), (2, 2, 1), (2, 1, 2), (3, 3, 1)]
    run_test_case("Tall and Thin", test_case_tall_thin, W=4, H=5, D=4)

    # test_case_flat_wide = [(1, 4, 4), (1, 3, 3), (1, 2, 2), (1, 1, 1), (2,1, 2)]
    # run_test_case("Wide and Flat", test_case_flat_wide, W=4, H=3, D=4)

    # test_case_tight_awkward = [(2, 2, 1), (1, 2, 2), (3, 1, 1), (1, 1, 3)]
    # run_test_case("Tight Awkward Fit", test_case_tight_awkward, W=3, H=3, D=3)
