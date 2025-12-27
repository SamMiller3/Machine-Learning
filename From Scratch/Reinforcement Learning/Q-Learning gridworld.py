from gymnasium import spaces
import numpy as np
import random

# 27/12/2025 Trivial gridworld environment using tabular Q-Learning epsilon greedy
# The world is a 3x3 NumPy array, the players position is denoted by a 1, the end state is denoted by a 2, empty space by 0
# the goal is at the bottom right, player starts top left

class MyEnv:
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3,3), dtype=np.int8)
        self.grid = np.array (
            [ 
                [1,0,0],
                [0,0,0],
                [0,0,2]
            ]
        , np.int32)
        self.agent_pos = [0,0]

    def reset(self, seed=None, options=None):
        self.agent_pos = [0,0]
        obs = self.get_obs()
        return obs, {}

    def step(self, action):
        if action == 0: # move up
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
        elif action == 1: # move down
            if self.agent_pos[0] < 2:
                self.agent_pos[0] += 1
        elif action == 2: # move right
            if self.agent_pos[1] < 2:
                self.agent_pos[1] += 1
        elif action == 3: # move left
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
        obs = self.get_obs()
        reward = 1.0 if self.agent_pos == [2,2] else -.01
        done = self.agent_pos == [2,2]
        return obs, reward, done
    
    def get_obs(self):
        grid = np.zeros((3, 3), dtype=np.int8)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1  # agent
        grid[2, 2] = 2  # goal
        row, column = self.agent_pos[0], self.agent_pos[1]
        return grid, row, column
    
# q learning 
state = 0
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
q_table = np.zeros((9, 4))
epochs = 15
env = MyEnv()

for epoch in range(epochs):
    cum_reward = 0
    obs, _ = env.reset()
    for i in range(100): # it gets 100 moves
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(q_table[state, :])
        
        obs, reward, done = env.step(action)
        _, row, column = env.get_obs()
        next_state = row * 3 + column
        q_table[state, action] = q_table[state,action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        state=next_state
        epsilon = max(0.01, epsilon * 0.99) # decay epsilon
        cum_reward+=reward
        if done:
            break
    print(epoch, cum_reward)
