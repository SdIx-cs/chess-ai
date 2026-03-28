import random
from extension.board_utils import list_legal_moves_for

def opponent(board, player, var):
    piece, move_opt = None, None
    
    if player.name == "white":

        while not move_opt:
            piece = random.choice(list(board.get_player_pieces(player)))
            mov = piece.get_move_options()
            if mov:
                move_opt = random.choice(mov)
                
                break
    else:
       
        while not move_opt:
            piece = random.choice(list(board.get_player_pieces(player)))
            mov = piece.get_move_options()
            if mov:
                move_opt = random.choice(mov)
                break
    
    return piece, move_opt
