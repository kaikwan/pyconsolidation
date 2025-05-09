# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from itertools import permutations, product
from scipy.spatial.transform import Rotation as R

# Setting the random seed for reproducibility
torch.manual_seed(42)
random.seed(42)


# Utility functions
def get_device(use_gpu=True):
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def compute_area(board, device):
    board_tensor = (
        torch.tensor(board, dtype=torch.int32, device=device) if not isinstance(board, torch.Tensor) else board
    )
    return int(torch.sum(board_tensor != 0).item())


def current_max_height(board, device):
    board_tensor = (
        torch.tensor(board, dtype=torch.int32, device=device) if not isinstance(board, torch.Tensor) else board
    )
    H = board_tensor.shape[0]
    for x in range(H - 1, -1, -1):
        if torch.any(board_tensor[x] != 0):
            return x + 1
    return 0


def bottom_fill_score(board, device):
    board_tensor = (
        torch.tensor(board, dtype=torch.int32, device=device) if not isinstance(board, torch.Tensor) else board
    )

    # If board_tensor is 4D, process it as a batch of boards
    if len(board_tensor.shape) == 4:  # Assuming shape is [batch_size, H, W, D]
        batch_size, H, W, D = board_tensor.shape
    elif len(board_tensor.shape) == 3:  # Single board
        H, W, D = board_tensor.shape
    else:
        raise ValueError(f"Unexpected tensor shape: {board_tensor.shape}")

    score = 0
    for x in range(H):
        filled = torch.sum(board_tensor[x] != 0).item()
        score += filled * (H - x - 1)

    return score


def compute_score(board, device):
    board_tensor = (
        torch.tensor(board, dtype=torch.int32, device=device) if not isinstance(board, torch.Tensor) else board
    )

    # Check the number of dimensions of the tensor
    if len(board_tensor.shape) == 3:
        # Single 3D board (H, W, D)
        H, W, D = board_tensor.shape
    elif len(board_tensor.shape) == 4:
        # Batch of 3D boards (batch_size, H, W, D)
        batch_size, H, W, D = board_tensor.shape
    else:
        raise ValueError(f"Unexpected tensor shape: {board_tensor.shape}, expected 3D or 4D tensor.")

    area = compute_area(board_tensor, device)
    # height = current_max_height(board_tensor, device)
    bottom_score = bottom_fill_score(board_tensor, device)

    # Compute floating holes (modified)
    floating_holes = 0
    for x in range(H):
        empty_cells = board_tensor[x] == 0
        if torch.any(empty_cells):
            for xx in range(x + 1, H):
                floating_holes += torch.sum((board_tensor[x] == 0) & (board_tensor[xx] != 0)).item()
                break

    return bottom_score + area * 2


def visualize_board_voxels(board):
    # If board is not a torch tensor, convert it
    board_tensor = board if isinstance(board, torch.Tensor) else torch.tensor(board)

    H, W, D = board_tensor.shape
    filled = board_tensor != 0  # Check where the board is not zero (filled)

    # Create an empty numpy array to hold the colors
    colors = np.empty((H, W, D), dtype=object)  # Use numpy object dtype to store colors

    # Map colors to the filled positions
    for x in range(H):
        for y in range(W):
            for z in range(D):
                if board_tensor[x, y, z] != 0:
                    # Convert the tensor value to a color using the colormap
                    colors[x, y, z] = plt.cm.tab20(
                        board_tensor[x, y, z].item() % 20
                    )  # Convert tensor to scalar for color mapping

    # Convert the filled and colors tensor to numpy for visualization
    filled_plot = filled.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]  # Change the axis order for visualization
    colors_plot = colors.transpose(1, 2, 0)[:, :, ::-1]

    # Plotting the voxels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled_plot, facecolors=colors_plot, edgecolor="k")
    ax.set_xlabel("Width (Y)")
    ax.set_ylabel("Depth (Z)")
    ax.set_zlabel("Height (X)")
    ax.set_xlim(0, W)
    ax.set_ylim(0, D)
    ax.set_zlim(H, 0)
    ax.set_xticks(torch.arange(0, W + 1, 1.0).numpy())  # Use torch.arange and convert to numpy for plotting
    ax.set_yticks(torch.arange(0, D + 1, 1.0).numpy())
    ax.set_zticks(torch.arange(0, H + 1, 1.0).numpy())

    max_range = max(W, D, H)
    ax.set_box_aspect([W / max_range, D / max_range, H / max_range])
    plt.tight_layout()
    plt.show()


def get_rotation_matrix(original_dims, rotated_dims):
    """
    Returns a 3x3 right-handed rotation matrix that maps original_dims to rotated_dims
    using permutations and sign flips. Assumes both are axis-aligned box dimensions.
    """
    orig = torch.tensor(original_dims, dtype=torch.float32)
    target = torch.tensor(rotated_dims, dtype=torch.float32)

    # Generate all valid rotation matrices: permutation * sign flip, right-handed only
    axes = torch.eye(3)

    for perm in permutations(range(3)):
        P = axes[list(perm)]  # permutation matrix

        for signs in product([-1, 1], repeat=3):
            R = P * torch.tensor(signs, dtype=torch.float32)  # apply sign flips

            if torch.isclose(torch.det(R), torch.tensor(1.0)):  # ensure right-handed
                R = R.to(orig.device)  # Ensure R is on the same device as orig
                transformed = torch.abs(R @ orig)  # Apply rotation to original dims

                # Check if the transformed result matches the target dims
                if torch.allclose(transformed, target.to(orig.device)):
                    return R

    raise ValueError(f"No valid rotation found from {original_dims} to {rotated_dims}")


def mat_to_quat(mat, flip=False):
    r = R.from_matrix(mat.cpu().numpy())
    if flip:
        # Adjust to 'zxy' order
        euler_deg = r.as_euler("xyz", degrees=True)
        r = R.from_euler("zxy", euler_deg, degrees=True)
    quat = r.as_quat(scalar_first=True)  # [w, x, y, z]
    return quat


def get_place_poses_from_placements(placed_blocks, id_to_index, blocks, flip=False):
    place_poses = {}

    for place_index, (block_index, placed_dims, (x, y, z), block_id) in enumerate(placed_blocks):
        original_dims = blocks[block_index]

        # Attempt to match the rotation (could also skip this if you trust placed_dims)
        try:
            rotation_matrix = get_rotation_matrix(original_dims, placed_dims)
            quat = mat_to_quat(rotation_matrix, flip)
        except ValueError:
            rotation_matrix = torch.eye(3)  # fallback: no valid rotation found
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity quaternion

        place_poses[place_index] = {
            "block_index": block_index,
            "position": (y, z, x) if flip else (x, y, z),
            "rotated_dims": (placed_dims[1], placed_dims[2], placed_dims[0]),
            "rotation_quat": quat,
        }

    return place_poses


def place_blocks_batched(boards, xs, ys, zs, h, w, d, block_ids):
    """
    Place multiple blocks in a batch of boards at given positions.
    Assumes all blocks have the same shape (h, w, d).
    """
    for i in range(boards.shape[0]):
        x, y, z = xs[i], ys[i], zs[i]
        boards[i, x : x + h, y : y + w, z : z + d] = block_ids[i]


def reconstruct_board(placed_blocks, H, W, D, device):
    board = torch.zeros((H, W, D), dtype=torch.int32, device=device)
    for block_index, dims, (x, y, z), block_id in placed_blocks:
        h, w, d = dims
        board[x : x + h, y : y + w, z : z + d] = block_id
    return board


def monte_carlo_beam_search(
    blocks, W=5, H=5, D=5, allow_rotations=False, beam_width=10, sample_size=10, trials=5, use_gpu=True
):
    device = get_device(use_gpu)

    def reconstruct_board(placed_blocks):
        board = torch.zeros((H, W, D), dtype=torch.int32, device=device)
        for _, dims, (x, y, z), block_id in placed_blocks:
            h, w, d = dims
            board[x : x + h, y : y + w, z : z + d] = block_id
        return board

    def can_place(board, x, y, z, h, w, d):
        if x + h > H or y + w > W or z + d > D:
            return False
        return torch.all(board[x : x + h, y : y + w, z : z + d] == 0)

    def is_grounded(board, x, y, z, h, w, d):
        if x == 0:
            return True
        region_below = board[x - 1, y : y + w, z : z + d]
        return torch.all((board[x, y : y + w, z : z + d] == 0) | (region_below != 0))

    def is_clear_above(board, x, y, z, h, w, d):
        """
        Returns True if all cells above the proposed block placement are empty.
        Prevents placing under overhangs or floating blocks.
        """
        # Check every cell in the proposed placement volume to ensure
        # there's no non-zero voxel above it (from x+h up to H)
        return all(not torch.any(board[xx, y : y + w, z : z + d] != 0) for xx in range(x + h, board.shape[0]))

    def is_x_order_valid(board, x, y, z, h, w, d):
        """
        Ensure that no part of the block is placed above an empty space in the layer below.
        Enforces strict bottom-up stacking.
        """
        if x == 0:
            return True  # Ground level is always valid

        below = board[x - 1 : x, y : y + w, z : z + d]
        current = board[x : x + 1, y : y + w, z : z + d]

        # Check for any non-zero in the current layer that's above a zero in the layer below
        return torch.all((current == 0) | (below != 0))

    def find_lowest_grounded_x(board, y, z, h, w, d):
        for x in range(H - h + 1):
            if (
                can_place(board, x, y, z, h, w, d)
                and is_grounded(board, x, y, z, h, w, d)
                and is_clear_above(board, x, y, z, h, w, d)
                and is_x_order_valid(board, x, y, z, h, w, d)
            ):
                return x
        return -1

    def compute_score_batch(boards):
        B, H, W, D = boards.shape
        filled = boards != 0
        area = filled.view(B, -1).sum(dim=1)
        weights = torch.arange(H - 1, -1, -1, device=boards.device).view(1, H, 1, 1)
        bottom_score = (filled * weights).sum(dim=(1, 2, 3))
        return bottom_score + area * 2

    best_score = -float("inf")
    best_placed_blocks = []
    best_id_to_index = {}

    for _ in range(trials):
        beam = [(0, [], list(enumerate(blocks)), 1, {})]  # score, placed_blocks, remaining, next_id, id_map

        while beam:
            candidates = []

            for score, placed_blocks, remaining, next_id, id_map in beam:
                board = reconstruct_board(placed_blocks)

                for i, (block_index, block) in enumerate(remaining):
                    rotations = list(set(permutations(block))) if allow_rotations else [block]

                    for dims in rotations:
                        h, w, d = dims
                        yzs = random.sample(
                            [(y, z) for y in range(W - w + 1) for z in range(D - d + 1)],
                            min(sample_size, (W - w + 1) * (D - d + 1)),
                        )

                        for y, z in yzs:
                            x = find_lowest_grounded_x(board, y, z, h, w, d)
                            if x == -1:
                                continue
                            new_placed = placed_blocks + [(block_index, dims, (x, y, z), next_id)]
                            new_remaining = remaining[:i] + remaining[i + 1 :]
                            new_id_map = id_map.copy()
                            new_id_map[next_id] = block_index
                            candidates.append((new_placed, new_remaining, next_id + 1, new_id_map))

            if not candidates:
                break

            # Reconstruct boards and compute scores in batch
            boards = torch.stack([reconstruct_board(p) for p, _, _, _ in candidates])
            scores = compute_score_batch(boards)

            # Get top-k indices
            topk_scores, topk_indices = torch.topk(scores, min(beam_width, scores.shape[0]))

            # Update beam
            beam = []
            for s, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
                placed, remaining, next_id, id_map = candidates[idx]

                if s > best_score:
                    best_score = s
                    best_placed_blocks = placed
                    best_id_to_index = id_map

                beam.append((s, placed, remaining, next_id, id_map))

    final_board = reconstruct_board(best_placed_blocks)
    return final_board, best_placed_blocks, best_id_to_index
