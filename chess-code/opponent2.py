# agent9
import random
import time
from extension.board_utils import list_legal_moves_for, copy_piece_move
from extension.board_rules import _position_key

# Common evaluation parameters
center_squares = [(2, 2)]          
extended_center = [(1,1), (1,2), (1,3), 
                    (2,1), (2,3),
                    (3,1), (3,2), (3,3),]
piece_values = {'Pawn': 1, 'Knight': 3, 'Bishop': 3, 'Right': 5, 'Queen': 9, 'King': 0}

_GLOBAL_STATE = {
    'transposition_table': {},
    'move_history': [],
    'tt_hits': 0,
    'tt_stores': 0,
    'killer_moves': {}
}

def opponent(board, player, var):
    """
    Phase 4: Minimax with Alpha-Beta Pruning & Time Management
    - Searches as deep as possible within time limit
    - Uses iterative deepening
    - Time limit: 10 seconds per move (safe margin from 14s)
    """
    global _GLOBAL_STATE
    var = _GLOBAL_STATE
    legal = list_legal_moves_for(board, player)
    if not legal:
        return None, None
    
    # Reset stats for each move 
    var['tt_hits'] = 0
    var['tt_stores'] = 0
    
    TIME_LIMIT_PER_MOVE = 14.0  
    
    start_time = time.time()

    # Use iterative deepening with time constraint
    best_move = None
    
    for depth in range(1, 10):  
        elapsed = time.time() - start_time
        
        try:
            move = minimax_search(board, player, depth=depth, 
                                 move_history=var['move_history'],
                                 transposition_table=var['transposition_table'],
                                 time_limit=TIME_LIMIT_PER_MOVE * 0.9,  
                                 start_time=start_time,
                                 var=var)
            
            if move:
                best_move = move
                elapsed = time.time() - start_time
            else:
                print(f"  Depth {depth} returned no move")
                break
        except TimeoutError:
            print(f"  v9: Depth {depth-1}")
            break
    
    if best_move:
        piece, move = best_move
        
        # Record move in history
        move_key = (type(piece).__name__, piece.position.x, piece.position.y, move.position.x, move.position.y)
        var['move_history'].append(move_key)
        
        if len(var['move_history']) > 20: var['move_history'] = var['move_history'][-20:]
        
        return best_move
    else:
        print(f"[{player.name}] WARNING: Minimax failed, using random move")
        return random.choice(legal)


def minimax_search(board, player, depth, move_history=None, transposition_table=None, time_limit=10.0, start_time=None, var=None):
    """
    Find the best move using minimax algorithm with alpha-beta pruning
    Returns: (piece, move) tuple for the best move
    """
    if move_history is None:
        move_history = []
    if transposition_table is None:
        transposition_table = {}
    if start_time is None:
        start_time = time.time()
    
    legal_moves = list_legal_moves_for(board, player)
    if not legal_moves:
        return None
    
    # order moves -> better pruning
    ordered_moves = order_moves(board, legal_moves, player)
    
    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    moves_evaluated = 0
    
    for piece, move in ordered_moves:
        if time.time() - start_time > time_limit * 0.80:
            if best_move: return best_move
            raise TimeoutError()
        
        sim_board = board.clone()
        sim_board, sim_piece, sim_move = copy_piece_move(sim_board, piece, move)
        
        if not sim_piece or not sim_move:
            continue
        
        # Make the move on the cloned board, evaluate recursively by depth to see consequence and penalise repetition
        try:
            sim_piece.move(sim_move)
            score = minimax_alphabeta(sim_board, depth - 1, alpha, beta, False, player, transposition_table, time_limit, start_time, var)
            
            moves_evaluated += 1
        
            move_key = (type(piece).__name__, piece.position.x, piece.position.y, move.position.x, move.position.y)
            repetition_count = move_history.count(move_key)
            if repetition_count > 0:
                penalty = repetition_count * 100  
                score -= penalty
            
            if score > best_score:
                best_score = score
                best_move = (piece, move)
                alpha = max(alpha, score)
                
        except TimeoutError:
            if best_move: return best_move
            raise
        except Exception as e:
            continue 
    
    return best_move

# Find the equivalent piece and move in a cloned board
def copy_piece_move(board, piece, move):
    try:
        if piece and move:
            temp_piece = None
            for tp in board.get_player_pieces(piece.player):
                if type(tp) is type(piece) and tp.position == piece.position:
                    temp_piece = tp
                    break
            if temp_piece is None:
                return board, None, None
            
            dest = getattr(move, "position", None)
            temp_move = None
            for m in temp_piece.get_move_options():
                m_dest = getattr(m, "position", None)
                if m_dest == dest:
                    temp_move = m
                    return board, temp_piece, temp_move    
            return board, temp_piece, None
        else:
            return board, None, None
    except Exception:
        return board, None, None


def minimax_alphabeta(board, depth, alpha, beta, maximizing_player, original_player, transposition_table, time_limit=10.0, start_time=None, var=None):
    """
    Minimax algorithm with Alpha-Beta Pruning, Transposition Tables, and Time Management
    """
    if start_time and time.time() - start_time > time_limit * 0.85:  
        raise TimeoutError()
    
    # Base case: reached maximum depth or game over
    if depth == 0: return evaluate_position(board, original_player)
    
    position_hash = _position_key(board)
    
    # Check transposition table, skip if move was evaluated at equal or greater depth
    if position_hash in transposition_table:
        cached_depth, cached_score = transposition_table[position_hash]
        if cached_depth >= depth:
            if var:
                var['tt_hits'] = var.get('tt_hits', 0) + 1
            return cached_score
    
    # Find players in this board by name
    our_player = None
    opp_player = None
    for p in board.players:
        if p.name == original_player.name: our_player = p
        else: opp_player = p
    
    if not our_player or not opp_player:
        score = evaluate_position(board, original_player)
        transposition_table[position_hash] = (depth, score)
        if var:
            var['tt_stores'] = var.get('tt_stores', 0) + 1
        return score
    
    # opponent side considerations
    current_player = our_player if maximizing_player else opp_player
    legal_moves = list_legal_moves_for(board, current_player)
    legal_moves = order_moves(board, legal_moves, current_player)
    if not legal_moves:
        score = evaluate_position(board, original_player)
        transposition_table[position_hash] = (depth, score)
        if var:
            var['tt_stores'] = var.get('tt_stores', 0) + 1
        return score
    
    # agent's turn
    if maximizing_player:
        max_eval = float('-inf')
        
        for piece, move in legal_moves:
            sim_board = board.clone()
            sim_board, sim_piece, sim_move = copy_piece_move(sim_board, piece, move)
            
            if sim_piece and sim_move:
                try:
                    sim_piece.move(sim_move)
                    eval_score = minimax_alphabeta(sim_board, depth - 1, alpha, beta, False, original_player, 
                                                  transposition_table, time_limit, start_time, var)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break 
                except TimeoutError:
                    raise 
                except:
                    pass 
        
        result = max_eval if max_eval != float('-inf') else evaluate_position(board, original_player)
        transposition_table[position_hash] = (depth, result)
        if var:
            var['tt_stores'] = var.get('tt_stores', 0) + 1
        return result
    # Opponent's turn
    else:
        min_eval = float('inf')
        
        for piece, move in legal_moves:
            sim_board = board.clone()
            sim_board, sim_piece, sim_move = copy_piece_move(sim_board, piece, move)
            
            if sim_piece and sim_move:
                try:
                    sim_piece.move(sim_move)
                    eval_score = minimax_alphabeta(sim_board, depth - 1, alpha, beta, True, original_player, 
                                                  transposition_table, time_limit, start_time, var)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    
                    if beta <= alpha:
                        break  
                except TimeoutError:
                    raise 
                except:
                    pass 
        
        result = min_eval if min_eval != float('inf') else evaluate_position(board, original_player)
        transposition_table[position_hash] = (depth, result)
        if var:
            var['tt_stores'] = var.get('tt_stores', 0) + 1
        return result


def evaluate_position(board, player):
    """
    Evaluate the board position from player's perspective
    
    Considers:
    - Material balance (piece values)
    - Piece positioning (center control, development)
    - Pawn promotion threats
    - Checkmate detection
    - Opening safety
    """    
    score = 0
    
    # Find players for cloned boards simulation
    our_player = None
    opp_player = None
    
    for p in board.players:
        if p.name == player.name:
            our_player = p
        else:
            opp_player = p
    
    if not our_player or not opp_player:
        return 0
    
    # Check for checkmate/stalemate and mark penalty
    our_legal_moves = list_legal_moves_for(board, our_player)
    opp_legal_moves = list_legal_moves_for(board, opp_player)
    if not our_legal_moves: return -100000
    if not opp_legal_moves: return 100000
    
    for piece in board.get_player_pieces(our_player):
        piece_type = type(piece).__name__
        score += piece_values.get(piece_type, 0)
    
    for piece in board.get_player_pieces(opp_player):
        piece_type = type(piece).__name__
        score -= piece_values.get(piece_type, 0)
    
    score += evaluate_piece_positions(board, our_player, opp_player)
    score += evaluate_pawn_threats(board, our_player, opp_player)
    score += evaluate_endgame(board, our_player, opp_player)
    score += evaluate_opening_safety(board, our_player, opp_player)
    
    return score

# bonus for good piece placements
def evaluate_piece_positions(board, player, opponent):
    bonus = 0
    
    for piece in board.get_player_pieces(player):
        piece_type = type(piece).__name__
        pos = (piece.position.x, piece.position.y)
    
        if pos in center_squares:
            if piece_type in ['Knight', 'Bishop', 'Pawn']:
                bonus += 30
            else:
                bonus += 10
        elif pos in extended_center:
            if piece_type in ['Knight', 'Bishop', 'Pawn']:
                bonus += 10
        
        # Knight and Bishop development bonus (move off back rank)
        if piece_type in ['Knight', 'Bishop']:
            back_rank = 0 if player.name == "white" else 4
            if pos[0] != back_rank:
                bonus += 20
        
        # Pawn advancement bonus
        if piece_type == 'Pawn':
            if player.name == "white":
                bonus += pos[0] * 5  # Further forward = better
            else:
                bonus += (4 - pos[0]) * 5
    
    # Same evaluation for opponent (subtract their positional advantage)
    for piece in board.get_player_pieces(opponent):
        piece_type = type(piece).__name__
        pos = (piece.position.x, piece.position.y)
        
        if pos in center_squares:
            if piece_type in ['Knight', 'Bishop', 'Pawn']:
                bonus -= 30
            else:
                bonus -= 10
        elif pos in extended_center:
            if piece_type in ['Knight', 'Bishop', 'Pawn']:
                bonus -= 10
        
        if piece_type in ['Knight', 'Bishop']:
            back_rank = 0 if opponent.name == "white" else 4
            if pos[0] != back_rank:
                bonus -= 20
        
        if piece_type == 'Pawn':
            if opponent.name == "white":
                bonus -= pos[0] * 5
            else:
                bonus -= (4 - pos[0]) * 5
    
    return bonus



def evaluate_opening_safety(board, player, opponent):
    """
    Opening phase
    Utilised when it's still not endgame (total pieces >= 12)
    Penalises 
    - major pieces movement in opening phase
    - unfavourable trade"""
    penalty = 0
    
    our_pieces = list(board.get_player_pieces(player))
    opp_pieces = list(board.get_player_pieces(opponent))
    total_pieces = len(our_pieces) + len(opp_pieces)
    
    if total_pieces >= 12: 
        
        for piece in our_pieces:
            piece_type = type(piece).__name__
            pos = (piece.position.x, piece.position.y)
        
            if player.name == "white":
                back_rank = 4
                pawn_rank = 3
                danger_rank = 2  
            else:
                back_rank = 0
                pawn_rank = 1
                danger_rank = 2  
            
            if piece_type in ['Right', 'Queen']:
                if pos[0] != back_rank:
                    is_attacked_by_pawn = False
                    for opp_piece in opp_pieces:
                        if type(opp_piece).__name__ == 'Pawn':
                            opp_pos = (opp_piece.position.x, opp_piece.position.y)
                            if opponent.name == "white":
                                # White pawns attack diagonally upward (decreasing row); vice versa
                                if (opp_pos[0] - 1 == pos[0] and 
                                    abs(opp_pos[1] - pos[1]) == 1):
                                    is_attacked_by_pawn = True
                                    break
                            else:
                                if (opp_pos[0] + 1 == pos[0] and 
                                    abs(opp_pos[1] - pos[1]) == 1):
                                    is_attacked_by_pawn = True
                                    break
                    
                    if is_attacked_by_pawn:
                        if piece_type == 'Queen':
                            penalty -= 800  
                        else:  
                            penalty -= 400
                    else:
                        penalty -= 100  
            
            # Minor pieces (Knight/Bishop) should develop
            if piece_type in ['Knight', 'Bishop']:
                is_attacked_by_pawn = False
                for opp_piece in opp_pieces:
                    if type(opp_piece).__name__ == 'Pawn':
                        opp_pos = (opp_piece.position.x, opp_piece.position.y)
                        if opponent.name == "white":
                            if (opp_pos[0] - 1 == pos[0] and 
                                abs(opp_pos[1] - pos[1]) == 1):
                                is_attacked_by_pawn = True
                                break
                        else:
                            if (opp_pos[0] + 1 == pos[0] and 
                                abs(opp_pos[1] - pos[1]) == 1):
                                is_attacked_by_pawn = True
                                break
                
                if is_attacked_by_pawn:
                    penalty -= 250 
        
        # Opponent
        for piece in opp_pieces:
            piece_type = type(piece).__name__
            pos = (piece.position.x, piece.position.y)
            
            if opponent.name == "white":
                back_rank = 4
            else:
                back_rank = 0
            
            if piece_type in ['Right', 'Queen'] and pos[0] != back_rank:
                is_attacked_by_our_pawn = False
                for our_piece in our_pieces:
                    if type(our_piece).__name__ == 'Pawn':
                        our_pos = (our_piece.position.x, our_piece.position.y)
                        if player.name == "white":
                            if (our_pos[0] - 1 == pos[0] and 
                                abs(our_pos[1] - pos[1]) == 1):
                                is_attacked_by_our_pawn = True
                                break
                        else:
                            if (our_pos[0] + 1 == pos[0] and 
                                abs(our_pos[1] - pos[1]) == 1):
                                is_attacked_by_our_pawn = True
                                break
                
                if is_attacked_by_our_pawn:
                    if piece_type == 'Queen':
                        penalty += 800
                    else:
                        penalty += 450
    return penalty



def evaluate_endgame(board, player, opponent):
    """
    Endgame phase
    if there's material advantage:
    - Push opponent king to corners
    - Bring the king closer to opponent king
    - Use pieces to restrict opponent king's movement
    """
    bonus = 0
    
    our_pieces = list(board.get_player_pieces(player))
    opp_pieces = list(board.get_player_pieces(opponent))
    
    our_material = sum(piece_values.get(type(p).__name__, 0) for p in our_pieces)
    opp_material = sum(piece_values.get(type(p).__name__, 0) for p in opp_pieces)
    
    total_pieces = len(our_pieces) + len(opp_pieces)
    material_advantage = our_material - opp_material
    
    if material_advantage >= 3 and total_pieces <= 8:
        our_king = None
        opp_king = None
        
        for piece in our_pieces:
            if type(piece).__name__ == 'King':
                our_king = piece
                break
        
        for piece in opp_pieces:
            if type(piece).__name__ == 'King':
                opp_king = piece
                break
        
        if our_king and opp_king:
            our_king_pos = (our_king.position.x, our_king.position.y)
            opp_king_pos = (opp_king.position.x, opp_king.position.y)
             
            corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
            min_corner_dist = min(abs(opp_king_pos[0] - c[0]) + abs(opp_king_pos[1] - c[1]) for c in corners)
            bonus += (8 - min_corner_dist) * 80  
            
            king_distance = abs(our_king_pos[0] - opp_king_pos[0]) + abs(our_king_pos[1] - opp_king_pos[1])
            
            # Ideal distance is 2 for opposition in checkmate
            if king_distance == 2:
                bonus += 100  
            elif king_distance == 3:
                bonus += 50
            else:
                bonus += max(0, (6 - king_distance) * 20)
            
            opp_king_moves = len([m for p, m in list_legal_moves_for(board, opponent) if type(p).__name__ == 'King'])
            
            if opp_king_moves == 0: pass # Checkmate or stalemate - handled elsewhere
            elif opp_king_moves == 1: bonus += 200  
            elif opp_king_moves == 2: bonus += 100
            else: bonus += (8 - opp_king_moves) * 30
            
            for piece in our_pieces:
                if type(piece).__name__ == 'Right':
                    piece_pos = (piece.position.x, piece.position.y)
                    if piece_pos[0] == opp_king_pos[0] or piece_pos[1] == opp_king_pos[1]:
                        bonus += 60  
    
    return bonus


def evaluate_pawn_threats(board, player, opponent):
    """
    Evaluate pawn promotion threats
    """
    bonus = 0
    
    for piece in board.get_player_pieces(player):
        if type(piece).__name__ == 'Pawn':
            pos = piece.position
            if player.name == "white":
                distance_to_promotion = pos.x
                if distance_to_promotion == 1:
                    bonus += 300  
                elif distance_to_promotion == 2:
                    bonus += 150
                elif distance_to_promotion == 3:
                    bonus += 50
            else:
                distance_to_promotion = 4 - pos.x
                if distance_to_promotion == 1:
                    bonus += 300
                elif distance_to_promotion == 2:
                    bonus += 150
                elif distance_to_promotion == 3:
                    bonus += 50
    
    for piece in board.get_player_pieces(opponent):
        if type(piece).__name__ == 'Pawn':
            pos = piece.position
            if opponent.name == "white":
                distance_to_promotion = pos.x
                if distance_to_promotion == 1:
                    bonus -= 400  
                elif distance_to_promotion == 2:
                    bonus -= 200
                elif distance_to_promotion == 3:
                    bonus -= 80
            else:
                distance_to_promotion = 4 - pos.x
                if distance_to_promotion == 1:
                    bonus -= 400  
                elif distance_to_promotion == 2:
                    bonus -= 200
                elif distance_to_promotion == 3:
                    bonus -= 80
    return bonus

# find pieces given move's target position (move is MoveOption object with move.position.x and move.position.y)
def get_piece_at(board, move): 
    target_pos = move.position if hasattr(move, 'position') else move
    
    for piece in board.get_pieces():
        if piece.position.x == target_pos.x and piece.position.y == target_pos.y:
            return piece
    return None

# Get the opponent player
def get_opponent(board, player):
    for p in board.players:
        if p != player: return p
    return None


def order_moves(board, legal_moves, player):
    """
    Order list of move from legal_move to improve minimax performance

    Move ordering priority:
    1. Checks 
    2. Captures of high-value pieces (Queen > Right > Bishop/Knight > Pawn)
    3. Good trades 
    4. Center control
    5. Other moves
    """
    
    scored_moves = []
    
    for piece, move in legal_moves:
        score = 0
        piece_type = type(piece).__name__
        attacker_value = piece_values.get(piece_type, 0)
        
        target = get_piece_at(board, move)
        
        if target and target.player != player:
            target_type = type(target).__name__
            captured_value = piece_values.get(target_type, 0)
            
            score += captured_value * 1000 
            score += (captured_value - attacker_value) * 100
            
            if attacker_value > captured_value:
                score -= 50  
        
        else:
            pos = (move.position.x, move.position.y)
            if pos in center_squares:
                score += 30
            if pos in extended_center:
                score += 15
            

            if piece_type in ['Knight', 'Bishop']:
                back_rank = 4 if player.name == "white" else 0
                if piece.position.x == back_rank and pos[0] != back_rank:
                    score += 25  
        
        scored_moves.append((piece, move, score))
    
    # Sort: highest first
    scored_moves.sort(key=lambda x: x[2], reverse=True)
    return [(piece, move) for piece, move, score in scored_moves]