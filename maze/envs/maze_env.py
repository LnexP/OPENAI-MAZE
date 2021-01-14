import os, subprocess, time, signal
import numpy as np
from .maze_ import Maze
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

class MazeEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    ACTION = ['N', 'S', 'E', 'W']

    def __init__(self, maze_size=None, maze_file=None, obstacle_num=4, static_goal=True, enable_render=True):
        self.steps = 0
        self.viewer = None
        self.static_goal = static_goal
        self.enable_render = enable_render
        self.maze_file = maze_file
        
        self.seed()
        self.maze = Maze(maze_size=maze_size, maze_file=maze_file, obstacle_num=obstacle_num, seed=self.seed)
        self.maze_size = self.maze.maze_size
        self.avaliable_area = np.where(self.maze.map != 1)
        self.len_area = self.avaliable_area[0].shape[0]
        print(maze_size)
        if maze_size == None and maze_file == None:
            raise AttributeError("A maze_file or maze_size must be set for the env")
        

        if not isinstance(obstacle_num, int):
            raise TypeError("obstacle num must be a positive integer!")
        elif obstacle_num <= 0:
            raise ValueError("obstacle num must be a positive integer!")


        self.action_space = spaces.Discrete(2*len(self.maze_size))
        self.observation_space = spaces.MultiDiscrete(tuple(self.maze_size)+tuple(self.maze_size))


        self.state = None
        self.steps_beyond_done = None

        print('-'*9 + 'maze env initialized successfully' + '-'*9)
        print('static goal:{}'.format(static_goal))
        print('map size:{}, obs num:{}'.format(self.maze.maze_size, self.maze.obs_num))
        print('cat position: {}'.format(self.maze.cat_pos))
        print('rat position: {}'.format(self.maze.rat_pos))
        
        # self.reset()

    def seed(self, seed=None):
        self.rng, self.seed = seeding.np_random(seed)
        self.seed = self.seed % (2**32 - 1)
        return self.seed

    def step(self, action):
        
        
        reward = self.maze.cat_move(action)
        done = False
        rat_action = -1
        if np.array_equal(self.maze.cat_pos, self.maze.rat_pos):
            # reward = 10*self.maze_size[0] * self.maze_size[1]
            reward = 10
            done = True

        if not done:
            if not self.static_goal:
                rat_action = self.maze.rat_move()

            if np.array_equal(self.maze.cat_pos, self.maze.rat_pos):
                # reward = 10 * np.prod(self.maze_size)
                reward = 10
                done = True

        if reward == -10:
            done == True
        # reward += (np.linalg.norm(self.maze_size) - np.linalg.norm(self.maze.cat_pos - self.maze.rat_pos, 2)) / np.linalg.norm(self.maze_size)
        info = {'cat':self.maze.cat_pos, 'rat':self.maze.rat_pos, 'rat_action':rat_action,'reward':reward}
        self.state = np.append(self.maze.cat_pos, self.maze.rat_pos)

        return self.state, reward, done, info

    def reset(self):
        # self.steps = (self.steps+3) % (len(self.avaliable_area[0]) ** 2)
        # if self.steps // self.len_area  == self.steps % self.len_area :
        #     self.steps += 1
        # self.maze.cat_pos = np.array([self.avaliable_area[0][self.steps // self.len_area ], self.avaliable_area[1][self.steps // self.len_area ]])
        # self.maze.rat_pos = np.array([self.avaliable_area[0][self.steps % self.len_area ], self.avaliable_area[1][self.steps % self.len_area ]])
        # print(self.steps)
        # print('cat position: {}'.format(self.maze.cat_pos))
        # print('rat position: {}'.format(self.maze.rat_pos))
        self.maze.generate_cat_pos()
        if not self.static_goal:
            self.maze.generate_rat_pos()
        return

    def close(self):
        pass

    def render(self, mode='human'):
        pass






