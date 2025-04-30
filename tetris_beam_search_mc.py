import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
import copy
from itertools import permutations

np.random.seed(42)
random.seed(42)

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

def monte_carlo_beam_search(blocks, W=5, H=5, D=5, allow_rotations=False, beam_width=10, sample_size=10, trials=5):

    def can_place(board, x, y, z, h, w, d):
        if x + h > H or y + w > W or z + d > D:
            return False
        return np.all(board[x:x+h, y:y+w, z:z+d] == 0)

    def is_grounded(board, x, y, z, h, w, d):
        if x == 0:
            return True
        return np.all((board[x, y:y+w, z:z+d] == 0) | (board[x-1, y:y+w, z:z+d] != 0))

    def place_block(board, x, y, z, h, w, d, block_id):
        board[x:x+h, y:y+w, z:z+d] = block_id
    def find_lowest_grounded_x(board, y, z, h, w, d):
        for x in range(H - h + 1):
            if can_place(board, x, y, z, h, w, d) and is_grounded(board, x, y, z, h, w, d):
                return x
        return None
    best_score = -float('inf')
    best_board = None

    for trial in range(trials):
        heap = [(0, 0, np.zeros((H, W, D), dtype=int), tuple(blocks))]
        state_counter = 1

        while heap:
            new_heap = []

            for _, _, board, remaining in heap:
                score = compute_score(board)
                if score > best_score:
                    best_score = score
                    best_board = board.copy()

                if not remaining:
                    continue

                idx = 0  # always expand first block in remaining
                block = remaining[idx]
                dims_list = [block]
                if allow_rotations:
                    dims_list = list(set(permutations(block)))

                for dims in dims_list:
                    h, w, d = dims
                    yzs = [(y, z) for y in range(W - w + 1) for z in range(D - d + 1)]
                    sampled_yzs = random.sample(yzs, min(sample_size, len(yzs)))

                    for y, z in sampled_yzs:
                        x = find_lowest_grounded_x(board, y, z, h, w, d)
                        if x is not None:
                            new_board = board.copy()
                            place_block(new_board, x, y, z, h, w, d, len(blocks) - len(remaining) + 1)
                            score = compute_score(new_board)
                            new_remaining = remaining[1:]
                            heapq.heappush(new_heap, (-score, state_counter, new_board, new_remaining))
                            state_counter += 1

            heap = heapq.nsmallest(beam_width, new_heap)

    return best_board

# Visualization utility for result
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
    ax.set_xlim(0, W)
    ax.set_ylim(0, D)
    ax.set_zlim(H, 0)
    ax.set_xticks(np.arange(0, W + 1, 1.0))
    ax.set_yticks(np.arange(0, D + 1, 1.0))
    ax.set_zticks(np.arange(0, H + 1, 1.0))

    # Set equal scaling for all axes
    max_range = max(W, D, H)
    ax.set_box_aspect([W / max_range, D / max_range, H / max_range])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    def run_test_case(name, blocks, W, H, D, allow_rotations=True, beam_width=5):
        print(f"\n=== Test Case: {name} ===")
        board = monte_carlo_beam_search(blocks, W=W, H=H, D=D, allow_rotations=allow_rotations, beam_width=beam_width, sample_size=20, trials=10)
        if board is not None:
            print(f"Best board found with score: {compute_score(board)}")
            print(f"Bottom fill score: {bottom_fill_score(board)}")
            print(f"Filled volume: {compute_area(board)}/{W*H*D}")
            print(f"Block Volume: {compute_area(board)}/{sum(h * w * d for h, w, d in blocks)}")
            print(f"Maximum height: {current_max_height(board)}/{H}")
            visualize_board_voxels(board)
        else:
            print("No valid board to visualize.")

    def generate_random_blocks(num_blocks, max_dim):
        return [(random.randint(1, max_dim), random.randint(1, max_dim), random.randint(1, max_dim)) for _ in range(num_blocks)]

    random_test_case = generate_random_blocks(num_blocks=20, max_dim=4)
    run_test_case("Random Test Case", random_test_case, W=10, H=5, D=5)
