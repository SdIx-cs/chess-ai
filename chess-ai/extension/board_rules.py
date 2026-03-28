from chessmaker.chess.pieces import King
from chessmaker.chess.results import no_kings, checkmate

# Creates a hashable key representing the current board position for repetition tracking
def _position_key(board): 
    pieces = []
    for p in board.get_pieces():
        pieces.append((p.name.lower(), getattr(p.player, "name", str(p.player)), p.position.x, p.position.y)) # e.g. ("pawn", "white", 0, 1)
    pieces.sort()
    side_to_move = getattr(board.current_player, "name", str(board.current_player))
    return (tuple(pieces), side_to_move)

# Updates and returns the repetition count for the current board position
def _update_repetition_count(board): 
    if not hasattr(board, "_rep_hist") or board._rep_hist is None:
        board._rep_hist = {}
    key = _position_key(board)
    if key in board._rep_hist:
        board._rep_hist[key] = board._rep_hist[key] + 1
    else:
        board._rep_hist[key] = 1    
        
    return board._rep_hist[key]

# Determines if the game has ended and returns the result string if so
def get_result(board): 
    rep_count = _update_repetition_count(board)
    if rep_count >= 5:
        return "Draw - fivefold repetition"
    for rf in [no_kings, checkmate, cannot_move, only_2kings]:
        res = rf(board)
        if res:
            return res
    return None

# Checks for stalemate (no legal moves for current player)
def cannot_move(board):  
    current_player = board.current_player
    player_pieces = list(board.get_player_pieces(current_player))
    for piece in player_pieces:
        if piece.get_move_options():
            return None

    return f"Stalemate (no more possible moves) - {current_player.name} loses" 

def only_2kings(board): 
    all_pieces = []
    for piece in board.get_pieces():
        all_pieces.append(piece)
        if len(all_pieces)>2:
            return None
    if len(all_pieces) == 2 and all(isinstance(piece, King) for piece in all_pieces):
        return "Draw - only 2 kings left"
