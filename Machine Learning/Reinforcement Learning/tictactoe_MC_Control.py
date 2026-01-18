# 18/01/26 TicTacToe environment and On-policy first-visit MC control (for ε-soft policies) agent learns through selfplay
# Sparse rewards as only +1 on win and -1 upon loss, 0 upon tie
# Uses a self relative board, so flips board when AI player 2 as 
# TicTacToe is symmetric and a zero sum game.

import numpy as np
import random

class MyEnv:
    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1
    
    def reset(self):
        self.board = [0] * 9
        obs = self.get_obs()
        self.current_player = 1 # 1 is Player 1 (X), -1 is Player 2 (O)
        return obs, {}
    
    def step(self, action):
        done = False
        reward = 0
        if self.board[action] == 0:
            self.board[action] = 1 if self.current_player == 1 else 2
            if self.check_win(1 if self.current_player == 1 else 2):
                reward = 1
                done = True
            elif self.game_over():
                done = True
            self.current_player *= -1
        return self.board, reward, done
    
    def game_over(self):
        return all(cell != 0 for cell in self.board)
    
    def check_win(self, player):
        lines = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]
        for line in lines:
            if all(self.board[pos] == player for pos in line):
                return True
        return False
    
    def get_obs(self):
        return self.board
    
    def get_self_relative(self):
        if self.current_player == 1:
            return self.board
        else:
            return [
                0 if cell == 0 else
                1 if cell == 2 else
                2
                for cell in self.board
            ]
    
    def state_to_int(self, board):
        state = 0
        for i, cell in enumerate(board):
            state += cell * (3**i)
        return state

learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
q_table = np.zeros((3**9, 9))
returns_count = np.zeros((3**9, 9), dtype=np.int32)
returns_sum = np.zeros((3**9, 9), dtype=np.float64)
epochs = int(input("How many epochs: "))
cum_reward = 0
env = MyEnv()

for epoch in range(epochs):
    obs, _ = env.reset()
    episode = [] # list of (state, action, player_sign)
    for i in range(9):
        current_player = env.current_player
        board = env.get_self_relative()
        state = env.state_to_int(board)
        valid_actions = [i for i in range(9) if board[i] == 0]
        
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            q_values = q_table[state, valid_actions]
            action = valid_actions[np.argmax(q_values)]
        
        episode.append((state, action, current_player))
        
        obs, reward, done = env.step(action)

        if done:
                if reward == 1:
                    winner = current_player
                else:
                    winner = 0 
                break
    epsilon = max(0.001, epsilon * 0.9995)
    seen = set()
    for (s,a,player) in episode:
        key = (s,a)
        if key in seen:
            continue
        seen.add(key)
        if winner == 0:
            G = 0
        else:
            G = 1.0 if player == winner else -1.0
        returns_count[s,a] += 1
        returns_sum[s,a] += G
        q_table[s, a] = returns_sum[s, a] / returns_count[s, a]
    
    if epoch % 30000 == 0:
        print("Epoch: ", epoch)

print(f"{epochs} of training complete")

print("=======================")
rematch = ""
while rematch.lower() != "n":
    obs, _ = env.reset()
    player = int(input("Do you want to be player 1 or 2? (1 or 2): "))
    print(" ")
    board_display = np.array(obs).reshape(3,3)
    board_display = np.where(board_display == '0', '-', board_display)
    print(np.array(obs).reshape(3,3))
    while True:
        if player == 1:
            move = int(input("enter move: "))-1
            obs, reward, done = env.step(move)
            board_display = np.array(obs).reshape(3,3)
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
                break
            
            board = env.get_self_relative()
            state = env.state_to_int(board)
            valid_actions = [i for i in range(9) if board[i] == 0]
            q_values = q_table[state, valid_actions]
            action = valid_actions[np.argmax(q_values)]
            obs, reward, done = env.step(action)
            board_display = np.array(obs).reshape(3,3)
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
                break
        else:
            board = env.get_self_relative()
            state = env.state_to_int(board)
            valid_actions = [i for i in range(9) if board[i] == 0]
            q_values = q_table[state, valid_actions]
            action = valid_actions[np.argmax(q_values)]
            obs, reward, done = env.step(action)
            board_display = np.array(obs).reshape(3,3)
            board_display = np.where(board_display == 1, 'X', board_display)
            board_display = np.where(board_display == 2, 'O', board_display)
            board_display = np.where(board_display == 0, '-', board_display)

            print(board_display)
            print(" ")
            if done:
                if reward == 1:
                    print("Player 1 (AI) wins!")
                else:
                    print("It was a draw!")
                print("=======================")
                break
            move = int(input("enter move: "))-1
            obs, reward, done = env.step(move)
            board_display = np.array(obs).reshape(3,3)
            board_display = np.where(board_display == 1, 'X', board_display)
            board_display = np.where(board_display == '2', 'O', board_display)
            board_display = np.where(board_display == '0', '-', board_display)
            print(board_display)
            print(" ")
            if done:
                if reward == 1:
                    print("Player 2 (human) wins!")
                else:
                    print("It was a draw!")
                print("=======================")
                break
        
    rematch = input("Press enter to rematch. Type n to quit: ")