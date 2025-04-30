import copy
from collections import deque

def can_place(board, x, y, h, w):
    H, W = len(board), len(board[0])
    if x + h > H or y + w > W:
        return False
    for i in range(h):
        for j in range(w):
            if board[x+i][y+j] != 0:
                return False
    return True

def place_block(board, x, y, h, w, block_id):
    new_board = [list(row) for row in board]
    for i in range(h):
        for j in range(w):
            new_board[x+i][y+j] = block_id
    return tuple(tuple(row) for row in new_board)

def is_bottom_filled(board):
    """Check that there are no floating blocks: no empty under filled cells"""
    H, W = len(board), len(board[0])
    for j in range(W):
        filled_found = False
        for i in range(H):
            if board[i][j] != 0:
                filled_found = True
            elif filled_found and board[i][j] == 0:
                return False  # hole under a block
    return True

def compute_area(board):
    return sum(cell != 0 for row in board for cell in row)

def board_priority(board):
    H = len(board)
    W = len(board[0])

    area = sum(cell != 0 for row in board for cell in row)
    bottom_score = 0
    for i, row in enumerate(board):
        filled = sum(cell != 0 for cell in row)
        bottom_score += filled * (H - i)  # rows near bottom get higher weight
    return (bottom_score, area)


def tetris_dp(blocks, W=10, H=20):
    initial_board = tuple(tuple(0 for _ in range(W)) for _ in range(H))
    initial_state = (initial_board, tuple(blocks))

    queue = deque()
    queue.append(initial_state)

    best_state = initial_state
    best_priority = board_priority(initial_board)

    seen = {}

    while queue:
        board, remaining = queue.popleft()

        if not remaining:
            current_priority = board_priority(board)
            if current_priority > best_priority:
                best_priority = current_priority
                best_state = (board, remaining)
            continue

        for idx, (h, w) in enumerate(remaining):
            best_placement = None
            best_x = best_y = best_width = None

            for (hh, ww) in [(h, w), (w, h)]:
                for y in range(W - ww + 1):  # for each column
                    for x in range(H - hh, -1, -1):  # from bottom to top
                        if can_place(board, x, y, hh, ww):
                            if (best_placement is None or
                                x > best_x or
                                (x == best_x and ww > best_width) or
                                (x == best_x and ww == best_width and y < best_y)):
                                best_placement = (x, y, hh, ww)
                                best_x, best_y, best_width = x, y, ww
                            break  # found lowest x for this y

            if best_placement is not None:
                x, y, hh, ww = best_placement
                block_id = len(blocks) - len(remaining) + 1
                new_board = place_block(board, x, y, hh, ww, block_id)
                if not is_bottom_filled(new_board):
                    continue

                new_remaining = remaining[:idx] + remaining[idx+1:]

                state = (new_board, new_remaining)
                state_priority = board_priority(new_board)

                if state not in seen or state_priority > seen[state]:
                    seen[state] = state_priority
                    queue.append((new_board, new_remaining))

                    if state_priority > best_priority:
                        best_priority = state_priority
                        best_state = (new_board, new_remaining)




    # Return the board configuration that achieved the best score and its area
    final_board, _ = best_state
    return final_board, compute_area(final_board)

def print_board(board):
    """
    Prints a visual representation of the board where:
    - Numbers represent placed blocks (block id)
    - Dots represent empty cells
    """
    for row in board:
        print(''.join(str(cell) if cell != 0 else '.' for cell in row))

# Example usage
# Each tuple (h, w) represents a block's height and width
blocks = [(7,2), (3,1), (5,1), (4,2), (3,3)]

# Run the tetris dynamic programming algorithm with width=10 and height=3
board, area = tetris_dp(blocks, W=10, H=3)

# Display the resulting board configuration
print_board(board)
# Show the total area filled by blocks
print(f"Total filled area: {area}")
