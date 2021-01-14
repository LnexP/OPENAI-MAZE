from maze.envs.maze_env import MazeEnv
import maze
import copy
import gym
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from collections import Counter
import time

def action_select(state, SIGMA):
    global q_table
    if np.random.random(1) < SIGMA:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

if __name__ == '__main__':
    kwargs = dict()
    kwargs['maze_file'] = './res/obstacles_4x4_obs_2.npy'
    # kwargs['obstacle_num'] = 5
    kwargs['static_goal'] = True
    env:MazeEnv = gym.make(id='maze-v0', **kwargs)
    ITER_MAX = np.prod(env.maze_size) *10
    SIGMA=0.0
    print(env.maze.obstacles)
    q_table = np.load('./2/q_table_4x4_obs_2_epoch_4999_GAMMA_0.9_SIGMA_0.02_LR_0.1_static_True.npy')
    env.reset()
    # env.maze.rat_pos = np.array([3,3])
    reward_all = 0
    
    env.maze.cat_pos = np.array([0, 0])
    env.maze.rat_pos = np.array([3,3])
    state = np.append(env.maze.cat_pos, env.maze.rat_pos)
    env.maze.GUI.start()
    time.sleep(5)
    env.maze.GUI.canvas.move('cat',env.maze.cat_pos[0]*25+25/2+25 ,env.maze.cat_pos[1]*25+25/2+25)
    env.maze.GUI.canvas.move('rat',env.maze.rat_pos[0]*25+25/2+25 ,env.maze.rat_pos[1]*25+25/2+25)
    
    for i in range(ITER_MAX):
        # print(env.maze.rat_pos)
        # print(i, state)
        action = action_select(tuple(list(state)), SIGMA)
        time.sleep(0.1)
        
        state_, reward, done, info = env.step(action)
        print(state_, state, action, info)
        env.maze.GUI.canvas.move('cat', 25*(state_[0] - state[0]), 25*(state_[1] - state[1]))
        time.sleep(0.1)
        # if (not done):
        env.maze.GUI.canvas.move('rat', 25*(state_[2] - state[2]), 25*(state_[3] - state[3]))
        # if np.array_equal(state_[0:2], state_[2:4]):
        #     print(_)
        reward_all += reward
        state = copy.copy(state_)

        # q_max = np.max(q_table[tuple(state_)])

        # q_table[tuple(np.append(state, action))] += LR * (reward + GAMMA * q_max - q_table[tuple(np.append(state, action))])

        # state = copy.copy(state_)
        if done:
            break