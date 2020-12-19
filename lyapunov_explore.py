#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
from itertools import permutations
from scipy.special import comb
import matplotlib
import matplotlib.pyplot as plt
import time
import csv
import multiprocessing

from grid_classes import *
from grid_utils import *

def lyapunov_exploration(w,kq,kQ,kZ,kY,seed_=0,show=False,savefigs=False):
    starttime = time.time()
    random.seed(seed_)
    np.random.seed(seed_)

    w.generate_failure_times()

    print("STARTING GRID")
    w.show_world([kq,kQ,kZ,kY],printout=False,plot=show,save=savefigs)
    print("exploring...")

    none_moved = 0

    # log metrics as starting point
    w.make_metrics()
    w.log_metrics()
    for r in w.get_robots():
        r.explore(r.state)
    for r in w.get_robots():
        r.share()

    T = 2
    while w.t < T:

        w.log_metrics()
        if show or savefigs:
            w.show_world([kq,kQ,kZ,kY],printout=False,plot=show,save=savefigs)

        if np.count_nonzero(w.base.grid) == w.base.grid.size:
            print("finished exploring!")
            break

        decisions = []
        for r in w.get_robots():
            if kq == 0 and kQ == 0 and kZ == 0 and kY == 0:
                best, best_score = r.unconstrained_optimize()
            elif kY == 0:
                ratio = kq
                best, best_score = r.time_preference_optimize(ratio)
            elif kq == -1 and kQ == -1 and kZ == -1 and kY == -1:
                best, best_score = r.constrained_optimize()
            elif kY == -1:
                best, best_score = r.multi_objective_optimize(kq,kQ,kZ,kY)
            else:
                best, best_score = r.lyapunov_optimize(kq,kQ,kZ,kY)
            decisions.append((r,best,best_score))

        for (r, best, best_score) in decisions:
            if best != r.state:
                r.move(best)
                none_moved = 0
            else:
                none_moved +=1
            if none_moved == len(w.robots):
                print("everyone is stuck")

        for r in w.get_robots():
            r.explore(r.state)
        for r in w.get_robots():
            r.share()

        running = w.increment_time()
        if not running:
            print("mission failed")
            break

    plt.close()
    print("ENDING GRID")
    w.show_world([kq,kQ,kZ,kY],printout=False,plot=show,save=savefigs)
    print("time",w.t)
    print(np.count_nonzero(w.base.grid), "cells explored")

    runtime = time.time() - starttime
    w.save_results(kq,kQ,kZ,kY,seed_,runtime)
    return

if __name__ == "__main__":

    world_name = sys.argv[1]
    config_filename = sys.argv[2]
    result_filename = config_filename
    # result_filename = sys.argv[3]

    progress = 1
    # for kq, kQ, kZ, kY in [[-1,-1,-1,-1],
    #                       [0,0,0,1],
    #                       [0.75,0,0,0],[0.95,0,0,0],
    #                       [0.0001,0,0,500],[0.001,0.001,0.001,1000],
    #                       [100,0,0,-1],[0.001,0.001,0.001,-1]]:
    for kq, kQ, kZ, kY in [[0.0001,0,0,500]]:
        processes = []
        # for seed_ in [0,1,2,3,4,5,6,7,8,9]:
        for seed_ in [0]:
            print("\nTrial:",progress)
            progress += 1

            if world_name == "baseline_world":
                w = World(30,30,[15,15])
                # 20 obstacles
                w.obstacles = [[22, 22], [11, 26], [28, 4], [11, 17], [0, 14], [11, 23], [23, 9], [13, 9], [14, 0], [22, 21], [19, 11], [4, 17], [12, 23], [20, 10], [11, 2], [24, 17], [17, 11], [26, 16], [23, 26], [14, 22]]
                w.obstacles.append([15,15])
                r1 = Robot(14,16,w)
                r2 = Robot(15,16,w)
                r3 = Robot(16,16,w)
                r4 = Robot(14,14,w)

            elif world_name == "more_area_world":
                w = World(50,50,[25,25])
                # 50 obstacles
                w.obstacles = [[20, 5], [28, 44], [18, 24], [22, 17], [44, 18], [33, 7], [24, 14], [46, 26], [41, 8], [4, 8], [34, 47], [19, 33], [29, 4], [48, 22], [35, 6], [7, 43], [12, 41], [1, 26], [13, 14], [46, 45], [49, 4], [12, 27], [15, 0], [31, 27], [17, 12], [13, 22], [41, 37], [20, 5], [30, 36], [49, 27], [41, 25], [37, 2], [12, 45], [29, 23], [43, 30], [20, 44], [33, 33], [47, 9], [20, 48], [37, 8], [14, 13], [6, 23], [41, 49], [19, 5], [15, 48], [2, 20], [5, 2], [35, 33], [4, 30], [32, 29]]
                w.targets = []
                w.obstacles.append([25,25])
                r1 = Robot(25,26,w)
                r2 = Robot(24,26,w)
                r3 = Robot(25,27,w)
                r4 = Robot(24,27,w)

            elif world_name == "more_obstacles_world":
                w = World(30,30,[15,15])
                # 75 obstacles
                w.obstacles = [[6, 14], [28, 27], [9, 7], [26, 27], [17, 15], [9, 18], [16, 7], [9, 1], [26, 1], [5, 4], [24, 9], [16, 5], [23, 14], [16, 17], [2, 3], [19, 19], [3, 20], [2, 6], [9, 7], [13, 24], [9, 25], [13, 0], [15, 27], [6, 21], [7, 28], [18, 11], [2, 10], [9, 14], [23, 11], [0, 16], [21, 1], [18, 19], [16, 3], [8, 23], [12, 9], [13, 3], [7, 24], [17, 6], [14, 7], [13, 22], [11, 20], [15, 25], [29, 12], [29, 23], [28, 3], [14, 8], [26, 23], [4, 22], [12, 14], [12, 15], [15, 1], [18, 14], [26, 7], [19, 7], [22, 14], [9, 24], [7, 25], [27, 16], [26, 29], [9, 9], [17, 6], [12, 24], [26, 19], [13, 6], [3, 27], [11, 6], [11, 20], [21, 5], [27, 0], [20, 29], [10, 12], [12, 28], [7, 23], [8, 13], [18, 26]]
                # w.obstacles = np.random.randint(0,30,(75,2)).tolist()
                w.targets = []
                w.obstacles.append([15,15])
                r1 = Robot(14,16,w)
                r2 = Robot(15,16,w)
                r3 = Robot(16,16,w)
                r4 = Robot(14,14,w)

            elif world_name == "more_robots_world":
                w = World(30,30,[15,15])
                # 20 obstacles
                w.obstacles = [[22, 22], [11, 26], [28, 4], [11, 17], [0, 14], [11, 23], [23, 9], [13, 9], [14, 0], [22, 21], [19, 11], [4, 17], [12, 23], [20, 10], [11, 2], [24, 17], [17, 11], [26, 16], [23, 26], [14, 22]]
                w.obstacles.append([15,15])
                r1 = Robot(14,14,w)
                r2 = Robot(14,15,w)
                r3 = Robot(14,16,w)
                r4 = Robot(15,14,w)
                r5 = Robot(15,16,w)
                r6 = Robot(16,14,w)
                r7 = Robot(16,15,w)
                r8 = Robot(16,16,w)

            else:
                print("That world is not defined in lyapunov_explore.py.")

            p = multiprocessing.Process(target=lyapunov_exploration, args=(w,kq,kQ,kZ,kY,seed_,False,True))
            # p = multiprocessing.Process(target=lyapunov_exploration, args=(w,kq,kQ,kZ,kY,seed_,True,False))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
