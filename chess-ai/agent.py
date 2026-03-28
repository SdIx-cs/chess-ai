import random
import time
from extension.board_utils import list_legal_moves_for, copy_piece_move
from extension.board_rules import _position_key


# Common values
center_squares = [(2, 2)]
extended_center = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)]
piece_values = {
    'Pawn': 100,
    'Knight': 320,
    'Bishop': 330,
    'Right': 500,
    'Queen': 900,
    'King': 20000
}

_GLOBAL_STATE = {
    'transposition_table': {},
    'move_history': [],
    'tt_hits': 0,
    'tt_stores': 0,
    'killer_moves': {}
}


def agent(board, player, var):
    global _GLOBAL_STATE
    var = _GLOBAL_STATE
    legal = list_legal_moves_for(board, player)
    if not legal:
        return None, None
    
    TIME_LIMIT_PER_MOVE = 13.5  # buffer
    start_time = time.time()
    
    best_move = None
    best_depth = 0
    
    # Iterative deepening
    for depth in range(1, 20):
        try:
            move = minimax_search(
                board, player, depth,
                var['move_history'],
                var['transposition_table'],
                TIME_LIMIT_PER_MOVE,
                start_time,
                var
            )
            
            if move:
                best_move = move
                best_depth = depth
            else:
                break
                
        except TimeoutError:
            break
    
    print(f"  Depth: {best_depth}")
    
    if best_move:
        piece, move_option = best_move
        
        # Record position hash => catches transpositions and move order differences
        sim_board, sim_piece, sim_move = copy_piece_move(board.clone(), piece, move_option)
        if sim_piece and sim_move:
            try:
                sim_piece.move(sim_move)
                position_hash = _position_key(sim_board)
                var['move_history'].append(position_hash)
                if len(var['move_history']) > 50:
                    var['move_history'] = var['move_history'][-50:]
            except:
                pass
        
        return best_move
    else:
        return random.choice(legal)
    

def get_players_fast(board, player):
    if board.players[0].name == player.name:
        return board.players[0], board.players[1]
    else:
        return board.players[1], board.players[0]


def minimax_search(board, player, depth, move_history, transposition_table, time_limit, start_time, var):
    """
    Minimax strategy
    1. Order moves
    2. Simulate moves with cloned board 
    3. Apply alphabeta pruning
    4. Penalise repeated moves
    """
    legal_moves = list_legal_moves_for(board, player)
    if not legal_moves: return None
    
    ordered_moves = order_moves_simple(board, legal_moves, player, depth, var) 
    
    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for piece, move_option in ordered_moves:
        if time.time() - start_time > time_limit:
            if best_move:
                return best_move
            raise TimeoutError()
        
        sim_board, sim_piece, sim_move = copy_piece_move(board.clone(), piece, move_option)
        
        if not sim_piece or not sim_move:
            continue
        
        try:
            sim_piece.move(sim_move)
            
            # Check stalemente
            sim_our_player = None
            for p in sim_board.players:
                if p.name == player.name:
                    sim_our_player = p
                    break
            
            if sim_our_player:
                future_our_moves = list_legal_moves_for(sim_board, sim_our_player)
                if not future_our_moves:
                    score = -50000
                else:
                    score = minimax_alphabeta(sim_board, depth - 1, alpha, beta, False, player, transposition_table, time_limit, start_time, var)
                    position_hash = _position_key(sim_board)
                    repetition_count = move_history.count(position_hash)
                    if repetition_count > 0:
                        score -= 50 * (2 ** repetition_count)
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_move = (piece, move_option)
                alpha = max(alpha, score)
                
        except TimeoutError:
            if best_move:
                return best_move
            raise
        except:
            continue
    
    return best_move


def minimax_alphabeta(board, depth, alpha, beta, maximizing_player, original_player, transposition_table, time_limit, start_time, var):
    """
    Alpha-beta pruning: 
    - move ordering
    - caching
    - killer moves
    - timeout handling"""
    if start_time and time.time() - start_time > time_limit: raise TimeoutError()
    if depth == 0: return evaluate_simple(board, original_player)
    
    position_hash = _position_key(board)
    if position_hash in transposition_table:
        cached_depth, cached_score = transposition_table[position_hash]
        if cached_depth >= depth:
            return cached_score
    
    # Find players
    our_player, opp_player = get_players_fast(board, original_player)
    
    if not our_player or not opp_player:
        return evaluate_simple(board, original_player)
    
    current_player = our_player if maximizing_player else opp_player
    legal_moves = list_legal_moves_for(board, current_player)
    
    if not legal_moves:
        return evaluate_simple(board, original_player)
    
    # Legal moves must be recalculated each board state, parent moves may no longer be valid; while transposition tables cache repeated positions for performance
    legal_moves = order_moves_simple(board, legal_moves, current_player, depth, var) 
    
    if maximizing_player:
        max_eval = float('-inf')
        for piece, move_option in legal_moves:
            sim_board, sim_piece, sim_move = copy_piece_move(board.clone(), piece, move_option)
            
            if sim_piece and sim_move:
                try:
                    sim_piece.move(sim_move)
                    eval_score = minimax_alphabeta(
                        sim_board, depth - 1, alpha, beta, False, original_player,
                        transposition_table, time_limit, start_time, var
                    )
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    
                    if beta <= alpha:
                        if depth not in var['killer_moves']:
                            var['killer_moves'][depth] = []
                        move_key = (piece.position.x, piece.position.y, move_option.position.x, move_option.position.y)
                        if move_key not in var['killer_moves'][depth]:
                            var['killer_moves'][depth].insert(0, move_key)
                            if len(var['killer_moves'][depth]) > 2:  # Keep only 2 killer moves per depth to save time
                                var['killer_moves'][depth] = var['killer_moves'][depth][:2]
                        break
                except TimeoutError:
                    pass
        
        result = max_eval if max_eval != float('-inf') else evaluate_simple(board, original_player)
    else:
        min_eval = float('inf')
        for piece, move_option in legal_moves:
            sim_board, sim_piece, sim_move = copy_piece_move(board.clone(), piece, move_option)
            
            if sim_piece and sim_move:
                try:
                    sim_piece.move(sim_move)
                    eval_score = minimax_alphabeta(
                        sim_board, depth - 1, alpha, beta, True, original_player,
                        transposition_table, time_limit, start_time, var
                    )
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    
                    if beta <= alpha:
                        break
                except TimeoutError:
                    raise
                except:
                    pass
        
        result = min_eval if min_eval != float('inf') else evaluate_simple(board, original_player)
    
    transposition_table[position_hash] = (depth, result)
    return result


def evaluate_simple(board, player):
    """
    Considerations
    - Material
    - position
    - pawn advancement
    - king activity (if endgame)
    - mobility
    """
    our_player, opp_player = get_players_fast(board, player)
    
    if not our_player or not opp_player:
        return 0
    
    our_moves = list_legal_moves_for(board, our_player)
    opp_moves = list_legal_moves_for(board, opp_player)
    if not our_moves: return -100000
    if not opp_moves: return 100000
    
    score = 0
     
    for piece in board.get_player_pieces(our_player): score += piece_values.get(type(piece).__name__, 0)
    
    for piece in board.get_player_pieces(opp_player): score -= piece_values.get(type(piece).__name__, 0)
    
    for piece in board.get_player_pieces(our_player):
        pos = (piece.position.x, piece.position.y)
        piece_type = type(piece).__name__
        
        if pos == (2, 2):
            if piece_type == 'Pawn':
                score += 50
            elif piece_type in ['Knight', 'Bishop']:
                score += 40
            else:
                score += 20
        
        elif (1 <= pos[0] <= 3 and 1 <= pos[1] <= 3):
            if piece_type == 'Pawn':
                score += 20
            elif piece_type in ['Knight', 'Bishop']:
                score += 15
    
    # Opponent
    for piece in board.get_player_pieces(opp_player):
        pos = (piece.position.x, piece.position.y)
        piece_type = type(piece).__name__
        
        if pos == (2, 2):
            if piece_type == 'Pawn':
                score -= 50
            elif piece_type in ['Knight', 'Bishop']:
                score -= 40
            else:
                score -= 20
        
        elif (1 <= pos[0] <= 3 and 1 <= pos[1] <= 3):
            if piece_type == 'Pawn':
                score -= 20
            elif piece_type in ['Knight', 'Bishop']:
                score -= 15
    
    for piece in board.get_player_pieces(our_player):
        if type(piece).__name__ == 'Pawn':
            if our_player.name == "white":
                score += (4 - piece.position.x) * 20
                if piece.position.x == 1:
                    score += 400  
            else:
                score += piece.position.x * 20
                if piece.position.x == 3:
                    score += 400
    
    for piece in board.get_player_pieces(opp_player):
        if type(piece).__name__ == 'Pawn':
            if opp_player.name == "white":
                score -= (4 - piece.position.x) * 20
                if piece.position.x == 1:
                    score -= 400  
            else:
                score -= piece.position.x * 20
                if piece.position.x == 3:
                    score -= 400
    
    total_pieces = len(list(board.get_pieces()))
    if total_pieces <= 6:
        score1, score2, score3 = 0, 0, 0
        our_king = None
        opp_king = None
        
        for piece in board.get_player_pieces(our_player):
            if type(piece).__name__ == 'King':
                our_king = piece
                break
        
        for piece in board.get_player_pieces(opp_player):
            if type(piece).__name__ == 'King':
                opp_king = piece
                break
        
        if our_king and opp_king:
            our_pos = (our_king.position.x, our_king.position.y)
            opp_pos = (opp_king.position.x, opp_king.position.y)
            
            edge_dist = min(opp_pos[0], 4 - opp_pos[0], opp_pos[1], 4 - opp_pos[1])
            score1 += (2 - edge_dist) * 200 
            
            king_dist = abs(our_pos[0] - opp_pos[0]) + abs(our_pos[1] - opp_pos[1])
            score2 += (8 - king_dist) * 50
            
            opp_move_count = len(opp_moves)
            if opp_move_count <= 5:
                score3 += (10 - opp_move_count) * 80
        score += (score1 + score2 + score3)*1.4
    
    score += (len(our_moves) - len(opp_moves)) * 10
    
    return int(score)


def order_moves_simple(board, legal_moves, player, depth, var):
    """
    Move ordering: 
    1. Killers 
    2. Captures 
    3. Center 
    4. Rest
    """
    killer_list = []
    captures = []
    center_moves = []
    other_moves = []
    killers = var.get('killer_moves', {}).get(depth, [])
    
    for piece, move_option in legal_moves:
        move_key = (piece.position.x, piece.position.y, move_option.position.x, move_option.position.y)
        
        if move_key in killers:
            killer_list.append((piece, move_option, 1000))
            continue
        
        target = get_piece_at(board, move_option)
        
        if target and target.player != player:
            captured_value = piece_values.get(type(target).__name__, 0)
            attacker_value = piece_values.get(type(piece).__name__, 0)
            score = captured_value * 10 - attacker_value
            captures.append((piece, move_option, score))
        
        elif (move_option.position.x, move_option.position.y) == (2, 2):
            center_moves.append((piece, move_option, 100))
        
        else:
            other_moves.append((piece, move_option, 0))
    
    captures.sort(key=lambda x: x[2], reverse=True)
    
    all_moves = killer_list + captures + center_moves + other_moves
    
    return [(piece, move_option) for piece, move_option, _ in all_moves]


def get_piece_at(board, move_option):
    """Find piece at target position"""
    target_pos = move_option.position if hasattr(move_option, 'position') else move_option
    
    for piece in board.get_pieces():
        if piece.position.x == target_pos.x and piece.position.y == target_pos.y:
            return piece
    return None