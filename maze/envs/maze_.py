import numpy as np
import copy
from skimage import measure
from collections import Counter
from .tk import MazeGUI
import re

class Maze:
    
    def __init__(self, maze_size=None, maze_file=None, obstacle_num=None, seed=1) -> None:

        self.obs_num = obstacle_num
        self.move = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]])
        self.seed = seed
        self.rat_pos = None
        self.cat_pos = None
        self.map = None
        if maze_size!=None:
            self.maze_size = np.array(maze_size)
            self.obstacles = None
            self.GUI = MazeGUI(self.obstacles, self.maze_size)
            self.initialize()
            print('-'*9 + 'maze map initialized successfully' + '-'*9)
        else:
            self.maze_file = maze_file
            m = re.search('[0-9]*x[0-9]*', maze_file)
            m = m[0].split('x')
            self.maze_size = [int(m[0]), int(m[1])]
            self.obstacles = list()
            mf = np.load(maze_file)
            self.map = np.zeros(self.maze_size, dtype=int)
            for i in range(mf.shape[0]):
                self.obstacles.append([mf[i, 0], mf[i, 1]])
                self.map[mf[i, 0], mf[i, 1]] = 1
            # self.obstacles = list(np.load(maze_file))
            
            # print(maze_file)
            
            self.GUI = MazeGUI(self.obstacles, self.maze_size)
            self.generate_pos()
            print('obs',self.obstacles)
            print('-'*9 + 'maze map loaded successfully' + '-'*9)
        
        

    
    def generate_map(self) -> bool:
        # print(self.seed)
        np.random.seed(self.seed)
        while True:
            self.obstacles = [[] for _ in range(self.obs_num)]
            self.map = np.zeros(self.maze_size)
            i = 0
            while i < self.obs_num:
                new_obs = [np.random.randint(low=0, high=self.maze_size[0]), np.random.randint(low=0, high=self.maze_size[1])]
                if new_obs in self.obstacles:
                    continue
                else:
                    self.obstacles[i] = new_obs
                    self.map[new_obs[0], new_obs[1]] = 1
                    i += 1
            
            # examine the connectivity of the map
            map_ = copy.deepcopy(self.map)
            map_[map_==0] = 2

            _ = measure.label(map_, connectivity=1)
            _ = Counter(_.flatten()).most_common(1)[0][1]
            
            if _ == self.maze_size[0] * self.maze_size[1] - self.obs_num:
                break

            del map_
        self.GUI = MazeGUI(self.obstacles, self.maze_size)
        return True

    def generate_pos(self) -> bool:

        while True:
            self.rat_pos = [np.random.randint(low=0, high=self.maze_size[0]), np.random.randint(low=0, high=self.maze_size[1])]
            if self.rat_pos not in self.obstacles:
                break

        while True:
            self.cat_pos = [np.random.randint(low=0, high=self.maze_size[0]), np.random.randint(low=0, high=self.maze_size[1])]
            if self.cat_pos not in self.obstacles and self.cat_pos != self.rat_pos:
                break
        self.cat_pos = np.array(self.cat_pos, dtype=int)
        self.rat_pos = np.array(self.rat_pos, dtype=int)
        return True

    def generate_cat_pos(self) -> bool:

        while True:
            self.cat_pos = [np.random.randint(low=0, high=self.maze_size[0]), np.random.randint(low=0, high=self.maze_size[1])]
            if self.cat_pos not in self.obstacles and self.cat_pos != list(self.rat_pos):
                break
        self.cat_pos = np.array(self.cat_pos, dtype=int)
        return True

    def generate_rat_pos(self) -> bool:

        while True:
            self.rat_pos = [np.random.randint(low=0, high=self.maze_size[0]), np.random.randint(low=0, high=self.maze_size[1])]
            if self.rat_pos not in self.obstacles and self.rat_pos != list(self.cat_pos):
                break
        self.rat_pos = np.array(self.rat_pos, dtype=int)
        return True

    def initialize(self):
        self.generate_map()
        self.generate_pos()

    
    def cat_move(self, action):
        
        cat_pos_ = self.cat_pos + self.move[action]
        # print(cat_pos_, sum(cat_pos_ > self.maze_size))
        if (cat_pos_ < 0).any():
            return -10
        elif (cat_pos_ >= self.maze_size).any():
            return -10
        elif self.map[cat_pos_[0], cat_pos_[1]] == 1:
            return -10
        else:
            self.cat_pos = copy.copy(cat_pos_)
            return -1

    def rat_move(self):
        while True:
            action = int(np.random.randint(0, 4, 1))
            # print(self.rat_pos.shape)
            rat_pos_ = self.rat_pos + self.move[action]
            # print(cat_pos_, sum(cat_pos_ > self.maze_size))
            if (rat_pos_ < 0).any():
                continue
            elif (rat_pos_ >= self.maze_size).any():
                continue
            elif self.map[rat_pos_[0], rat_pos_[1]] == 1:
                continue
            else:
                self.rat_pos = copy.copy(rat_pos_)
                return action


        
                

        

        

        