# prints board situation (Upper = white, lower = black) 
def print_board_ascii(board): 
    piece_map = {"pawn": "P", "right": "R", "knight": "N", "bishop": "B", "queen": "Q", "king": "K"}
    grid = [["." for _ in range(5)] for _ in range(5)]
    for piece in board.get_pieces():
        pos = piece.position
        ch = piece_map.get(piece.name.lower(), "?")
        grid[pos.y][pos.x] = ch.upper() if piece.player.name.lower() == "white" else ch.lower()
    print("  0 1 2 3 4")
    for row in (range(5)):
        print(f"{row} " + " ".join(grid[row]))

# Collects all legal moves a specific player can currently make, returning them as (piece, move) pairs
def list_legal_moves_for(board, player): 
    pairs = []
    for pc in board.get_player_pieces(player):
        for opt in pc.get_move_options():
            pairs.append((pc, opt))
    return pairs

# Clones board to simulate moves for evaluation without altering original board
def copy_piece_move(board, piece, move):  
    try:
        if piece and move:
            for tp in board.get_player_pieces(piece.player):
                if type(tp) is type(piece) and tp.position == piece.position:
                    temp_piece = tp
                    break
            if temp_piece is None:
                return board, None, None
            # find the equivalent move option on the cloned piece
            dest = getattr(move, "position", None)
            temp_move = None
            for m in temp_piece.get_move_options():
                m_dest = getattr(m, "position", None)
                if m_dest == dest:
                    temp_move = m
                    return board, temp_piece, temp_move # board = cloned board, temp_piece = cloned piece, temp_move = cloned move option (this can be reapplied to evaluate legality next hypothetical move (layering))
            return board, temp_piece, None 
        else:
            return board, None, None
    except Exception:
        return board, None, None
