from maze.envs.maze_env import MazeEnv
import maze
import copy
import gym
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from collections import Counter
import time
import argparse
import re


def action_select(state, SIGMA):
    if np.random.random(1) < SIGMA:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_size', nargs='+', type=int)
    parser.add_argument('--obstacle_num', type=int)
    parser.add_argument('--static_goal', default='False')
    parser.add_argument('--maze_file')
    parser.add_argument('--epoch', default=500000, type=int)
    parser.add_argument('--gamma', default=0.9999, type=float)
    parser.add_argument('--lr', default=0.8, type=float)
    parser.add_argument('--sigma', default=0.02, type=float)
    # parser.add_argument()
    args = parser.parse_args()
    kwargs = dict()
    if args.maze_size:
        kwargs['maze_size'] = list(args.maze_size)
        
    if args.obstacle_num:
        kwargs['obstacle_num'] = args.obstacle_num
        
    if args.static_goal != None:
        if args.static_goal == 'False':
            kwargs['static_goal'] = False
        else:
             kwargs['static_goal'] = True
        # print(args.static_goal)
        # print('-'*1000)
        # print('-'*1000)
        # print('-'*1000)
        # print(args.static_goal)
    if args.maze_file:
        kwargs['maze_file'] = args.maze_file
        m = re.search('[0-9]*x[0-9]*', args.maze_file)
        m = m[0].split('x')
        maze_size = [int(m[0]), int(m[1])]
        # kwargs['maze_size'] = maze_size
        o = np.load(args.maze_file)
        kwargs['obstacle_num'] = o.shape[0]
        

    env:MazeEnv = gym.make(id='maze-v0', **kwargs)
    # print(args.static_goal)
    # print(maze_size)
    # print(kwargs['obstacle_num'])
    # print('sigma:{}, gamma:{}, lr:{}'.format(args.sigma, args.gamma, args.lr))

    EPOCHS = args.epoch
    ITER_MAX = np.prod(env.maze_size) *10
    # ITER_MAX = 4000

    GAMMA = args.gamma
    
    SIGMA = args.sigma

    
    rewardss = np.zeros(EPOCHS)
    # print(q_table.shape)
    # q_table = np.zeros((env.maze_size[0], env.maze_size[1], env.action_space.n))
    # rewards = np.zeros(EPOCHS)
    
    rewards_mean = np.zeros(EPOCHS - 10)
    ss = time.time()
    env.maze.rat_pos = np.array([3,3])
    print('obs',env.maze.obstacles)
    for _ in range(1):
        # LR = args.lr
        rewards = []
        q_table = np.zeros(tuple(env.observation_space.nvec) + (env.action_space.n,))
        for e in range(EPOCHS):

            # SIGMA = SIGMA * 0.99 ** (e//100)
            LR = max(args.lr * 0.9995 ** (e//100), 0.1)
            # print(env.maze.cat_pos)
            env.reset()
            reward_all = 0
            state = np.append(env.maze.cat_pos, env.maze.rat_pos)
            # print(state)
            # dist = np.linalg.norm(state[0:2]-state[2:4], ord=1)
            # print(env.maze.rat_pos)
            # print(q_table[env.maze.rat_pos[0],env.maze.rat_pos[1]-2,:])
            s = time.time()
            for i in range(ITER_MAX):
                # print(env.maze.rat_pos)
                # print(i, state)
                action = action_select(tuple(list(state)), SIGMA)

                state_, reward, done, info = env.step(action)
                # dist_ = np.linalg.norm(state_[0:2]-state_[2:4], ord=1)
                # reward += (dist - dist_)*0.25
                # if np.array_equal(state_[0:2], state_[2:4]):
                #     print(_)
                reward_all += reward

                q_max = np.max(q_table[tuple(state_)])
                
                q_table[tuple(np.append(state, action))] += LR * (reward + GAMMA * q_max - q_table[tuple(np.append(state, action))])
                if done:
                    # print(info)
                    break
                state = copy.copy(state_)
                # dist = dist_
            if e >= 10 and e%1000 == 0:
                rewards_mean[e-10] = np.mean(rewards[e-10:e])
                # if 0.99<rewards_mean[e-10] / rewards_mean[e-1010]<1.01 and reward_all>maze_size[0]*maze_size[1]*9:
                #     break
                en = time.time()
                print('epoch:{:0>5d}, steps:{:0>5d}, current reward:{:>6}, spend time:{:7}, total time:{:7}'.format(e, i, np.mean(rewards[e-10:e]), en-s, en-ss))
            # if e % 100000 == 0:
            #     np.save('./res/q_table_{}x{}_obs_{}_epoch_{}_GAMMA_{}_SIGMA_{}_LR_{}_static_{}_transverse.npy'.format(maze_size[0], maze_size[1], kwargs['obstacle_num'], e, GAMMA, args.sigma, LR, kwargs['static_goal']), q_table)
            #     np.save('./res/rewards_{}x{}_obs_{}_epoch_{}_GAMMA_{}_SIGMA_{}_LR_{}_static_{}_transverse.npy'.format(maze_size[0], maze_size[1], kwargs['obstacle_num'], e, GAMMA, args.sigma, LR, kwargs['static_goal']), np.array(rewards))
            rewards.append(reward_all)
        rewardss += np.array(rewards)/20
        # print(q_table[:,:,3,3,:])
    np.save('./2/q_table_{}x{}_obs_{}_epoch_{}_GAMMA_{}_SIGMA_{}_LR_{}_static_{}.npy'.format(maze_size[0], maze_size[1], kwargs['obstacle_num'], e, GAMMA, args.sigma, LR, kwargs['static_goal']), q_table)
    np.save('./3/rewardss_{}x{}_obs_{}_epoch_{}_GAMMA_{}_SIGMA_{}_LR_{}_static_{}.npy'.format(maze_size[0], maze_size[1], kwargs['obstacle_num'], e, GAMMA, args.sigma, args.lr, kwargs['static_goal']), np.array(rewardss))
    # np.save('./res/obstacles_{}x{}_obs_{}.npy'.format(kwargs['maze_size'][0], kwargs['maze_size'][1], kwargs['obstacle_num']), np.array(env.maze.obstacles))
    # plt.plot(np.arange(10, EPOCHS), rewards_mean)
    # plt.show()