# 5/01/2025 MiniMax Noughts and Crosses solver

import numpy as np
import math

def init_board():
        board = [0] * 9
        return(board)
    
def step(action,board,player):
    done = False
    reward = 0
    if board[action] == 0:
        board[action] = 1 if player == 1 else 2
        if check_win(1 if player == 1 else 2, board):
            reward = 1
            done = True
        elif game_over(board):
            done = True
    return board, reward, done
    
def game_over(board):
    return all(cell != 0 for cell in board)
    
def check_win(player,board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in lines:
        if all(board[pos] == player for pos in line):
            return True
    return False

def state_to_int(self, board):
    state = 0
    for i, cell in enumerate(board):
        state += cell * (3**i)
    return state

def get_player_move(board,player):
    move = int(input("enter move: "))-1
    board, reward, done = step(move,board,player)
    board_display = np.array(board).reshape(3,3)
    board_display = np.where(board_display == 1, 'X', board_display)
    board_display = np.where(board_display == '2', 'O', board_display)
    board_display = np.where(board_display == '0', '-', board_display)
    print(board_display)
    print(" ")
    if done:
        if reward == 1:
            print("Player 1 (human) wins!")
        else:
            print("It was a draw!")
        print("=======================")
    return(board,done)
    
def ai_make_move(board,player):
    action = minimax(board,player,player)[1]
    board, reward, done = step(action,board,player)
    board_display = np.array(board).reshape(3,3)
    board_display = np.where(board_display == 1, 'X', board_display)
    board_display = np.where(board_display == '2', 'O', board_display)
    board_display = np.where(board_display == '0', '-', board_display)

    print(board_display)
    print(" ")
    if done:
        if reward == 1:
            print("Player 2 (AI) wins!")
        else:
            print("It was a draw!")
        print("=======================")
    return(board,done)

def minimax(board,cur_player,ai_player):
    board = board.copy()
    valid_actions = [i for i in range(9) if board[i] == 0]
    if ai_player == cur_player: # maximising
        best_val = -math.inf
        best_action = None
        for action in valid_actions:
            new_board, reward, done = step(action,board.copy(),cur_player)
            if done: # evaluate state
                val = reward
            else:
                val, _ = minimax(new_board,2 if cur_player == 1 else 1,ai_player)
            if val > best_val:
                best_val = val
                best_action = action
        return best_val, best_action
    else:
        best_val = math.inf
        best_action = None
        for action in valid_actions:
            new_board, reward, done = step(action,board.copy(),cur_player)
            if done: # evaluate state
                val = -reward
            else:
                val, _ = minimax(new_board,2 if cur_player == 1 else 1,ai_player)
            if val < best_val:
                best_val = val
                best_action = action
            
        return best_val, best_action
            



rematch = ""
while rematch.lower() != "n":
    board = init_board()
    player = int(input("Do you want to be player 1 or 2? (1 or 2): "))
    print(" ")
    board_display = np.array(board).reshape(3,3)
    board_display = np.where(board_display == 0, '-', board_display)
    print(board_display)
    while True:
            if player == 1:
                board,done = get_player_move(board,player)
                if done:
                    break
                board,done = ai_make_move(board,2 if player == 1 else 1)
                if done:
                    break
            else:
                board,done = ai_make_move(board,player)
                if done:
                    break
                board,done = get_player_move(board,2 if player == 1 else 1)
                if done:
                    break
