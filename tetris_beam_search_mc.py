import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
import copy
from itertools import permutations
import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    for x in range(H):
        empty_cells = board_np[x] == 0
        if np.any(empty_cells):
            for xx in range(x+1, H):
                floating_holes += np.sum((board_np[x] == 0) & (board_np[xx] != 0))
                break

    return bottom_score + area * 2 # - height - floating_holes

def is_grounded(board, x, y, z, h, w, d):
    if x == 0:
        return True
    return np.all((board[x, y:y+w, z:z+d] == 0) | (board[x-1, y:y+w, z:z+d] != 0))

def monte_carlo_beam_search(blocks, W=5, H=5, D=5, allow_rotations=False, beam_width=10, sample_size=10, trials=5, track_sequence=False):
    def can_place(board, x, y, z, h, w, d):
        if x + h > H or y + w > W or z + d > D:
            return False
        return np.all(board[x:x+h, y:y+w, z:z+d] == 0)

    def place_block(board, x, y, z, h, w, d, block_id):
        board[x:x+h, y:y+w, z:z+d] = block_id

    def find_lowest_grounded_x(board, y, z, h, w, d):
        for x in range(H - h + 1):
            if can_place(board, x, y, z, h, w, d) and is_grounded(board, x, y, z, h, w, d):
                return x
        return None

    best_score = -float('inf')
    best_board = None
    best_sequence = [] if track_sequence else None
    best_id_to_index = {}

    block_id_counter = 1

    initial_heap_item = (
        0, 0, np.zeros((H, W, D), dtype=int), tuple(enumerate(blocks)), [], {}
    ) if track_sequence else (
        0, 0, np.zeros((H, W, D), dtype=int), tuple(enumerate(blocks)), {}
    )

    heap = [initial_heap_item]
    state_counter = 1

    for trial in range(trials):
        local_heap = copy.deepcopy(heap)
        while local_heap:
            new_heap = []

            for item in local_heap:
                if track_sequence:
                    _, _, board, remaining, sequence, id_to_index = item
                else:
                    _, _, board, remaining, id_to_index = item
                    sequence = None

                score = compute_score(board)
                if score > best_score:
                    best_score = score
                    best_board = board.copy()
                    best_id_to_index = id_to_index.copy()
                    if track_sequence:
                        best_sequence = sequence.copy()

                if not remaining:
                    continue

                for i, (block_index, block) in enumerate(remaining):
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
                                block_id = block_id_counter
                                block_id_counter += 1

                                place_block(new_board, x, y, z, h, w, d, block_id)

                                # Remove used block
                                new_remaining = remaining[:i] + remaining[i+1:]
                                new_id_to_index = id_to_index.copy()
                                new_id_to_index[block_id] = block_index

                                if track_sequence:
                                    new_sequence = sequence + [new_board.copy()]
                                    heapq.heappush(new_heap, (-score, state_counter, new_board, new_remaining, new_sequence, new_id_to_index))
                                else:
                                    heapq.heappush(new_heap, (-score, state_counter, new_board, new_remaining, new_id_to_index))

                                state_counter += 1

            local_heap = heapq.nsmallest(beam_width, new_heap)

    if track_sequence:
        return best_board, best_sequence, best_id_to_index
    else:
        return best_board, best_id_to_index

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

    filled_plot = np.transpose(filled, (1, 2, 0))[:, :, ::-1]
    colors_plot = np.transpose(colors, (1, 2, 0))[:, :, ::-1]

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
    max_range = max(W, D, H)
    ax.set_box_aspect([W / max_range, D / max_range, H / max_range])
    plt.tight_layout()
    plt.show()

def save_voxel_animation(sequence, W, H, D, filename="animation.gif"):
    frames = []
    for board in sequence:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        board_np = np.array(board)
        filled = board_np != 0
        colors = np.empty(board_np.shape, dtype=object)

        for x in range(H):
            for y in range(W):
                for z in range(D):
                    if board_np[x, y, z] != 0:
                        colors[x, y, z] = plt.cm.tab20(board_np[x, y, z] % 20)

        filled_plot = np.transpose(filled, (1, 2, 0))[:, :, ::-1]
        colors_plot = np.transpose(colors, (1, 2, 0))[:, :, ::-1]

        ax.voxels(filled_plot, facecolors=colors_plot, edgecolor='k')
        ax.set_xlim(0, W)
        ax.set_ylim(0, D)
        ax.set_zlim(H, 0)
        ax.axis('off')
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.get_renderer().tostring_argb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, 1:]  # Drop the alpha channel to keep RGB format
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(filename, frames, duration=0.5)

if __name__ == "__main__":
    def run_test_case(name, blocks, W, H, D, allow_rotations=True, beam_width=25, save_animation=True):
        print(f"\n=== Test Case: {name} ===")
        board_and_seq = monte_carlo_beam_search(
            blocks, W=W, H=H, D=D,
            allow_rotations=allow_rotations,
            beam_width=beam_width,
            sample_size=25,
            trials=10,
            track_sequence=save_animation
        )
        if save_animation:
            board, sequence, id_to_index = board_and_seq
        else:
            board, id_to_index = board_and_seq

        if board is not None:
            print(f"Best board found with score: {compute_score(board)}")
            print(f"Bottom fill score: {bottom_fill_score(board)}")
            print(f"Filled volume: {compute_area(board)}/{W*H*D}")
            print(f"Block Volume: {compute_area(board)}/{sum(h * w * d for h, w, d in blocks)}")
            print(f"Maximum height: {current_max_height(board)}/{H}")
            placed_ids = sorted(set(int(i) for i in np.unique(board) if i > 0))
            placed_blocks = [blocks[id_to_index[i]] for i in placed_ids]
            remaining_blocks = [blocks[i] for i in range(len(blocks)) if i not in id_to_index.values()]

            print(f"Placed blocks: {placed_blocks}")
            print(f"Remaining blocks: {remaining_blocks}")
            print(f"Number of blocks placed: {len(placed_blocks)}")
            print(f"Number of blocks remaining: {len(remaining_blocks)}")
            if save_animation:
                gif_name = f"{name.replace(' ', '_')}.gif"
                save_voxel_animation(sequence, W, H, D, filename=gif_name)
                print(f"Animation saved as {gif_name}")
            visualize_board_voxels(board)
        else:
            print("No valid board to visualize.")

    def generate_random_blocks(num_blocks, max_dim):
        return [(random.randint(1, max_dim), random.randint(1, max_dim), random.randint(1, max_dim)) for _ in range(num_blocks)]

    random_test_case = generate_random_blocks(num_blocks=20, max_dim=8)
    run_test_case("Random Test Case", random_test_case, W=10, H=5, D=8, save_animation=True)
