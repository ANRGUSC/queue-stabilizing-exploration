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

def lyapunov_exploration(w,kq,kQ,kZ,kY,b,theta_fiedler,theta_CRB,show=False,shuffle=False,T=None, seed_=0):
    starttime = time.time()
    random.seed(seed_)
    np.random.seed(seed_)

    w.generate_failure_times()

    if T is None:
        T = LIFETIME

    print("STARTING GRID")
    w.show_world(printout=False,plot=show,save=True)
    print("exploring...")

    steps_taken = 0
    none_moved = 0

    # log metrics as starting point
    w.make_metrics()
    w.log_metrics()
    for r in w.get_robots():
        r.explore(r.state)
    for r in w.get_robots():
        r.share()

    while w.t < T:

        w.log_metrics()

        if show:
            w.show_world(printout=False,plot=True,save=False)
            if w.t % 10 == 0:
                plt.close()

        # show grid everytime
        # print(w.t)
        w.show_world(printout=False,plot=False,save=True)

        if np.count_nonzero(w.base.grid) == w.base.grid.size:
            print("finished exploring!")
            break

        if world_name == "lavatube_world" and np.count_nonzero(w.base.grid) == 113:
            print("finished exploring!")
            break

        decisions = []
        for r in w.get_robots():
            if kq == 0 and kQ == 0 and kZ == 0 and kY == 0:
                best, best_score = r.random_step()
            if kq == -1 and kQ == -1 and kZ == -1 and kY == -1:
                best, best_score = r.constrained_optimize(theta_fiedler,theta_CRB,shuffle)
            else:
                best, best_score = r.lyapunov_optimize(kq,kQ,kZ,kY,b,theta_fiedler,theta_CRB,shuffle)
            decisions.append((r,best,best_score))

        for (r, best, best_score) in decisions:
            if best != r.state:
                steps_taken += 1
                r.move(best)
                none_moved = 0
            else:
                none_moved +=1
            if none_moved == len(w.robots):
                print("stuck")

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
    w.show_world(printout=False,plot=show,save=True)
    print(steps_taken, "steps taken")
    print("time",w.t)
    print(np.count_nonzero(w.base.grid), "cells explored")

    # w.plot_metrics(kq, kQ, kZ, kY, seed_, show=False,save=False)
    runtime = time.time() - starttime
    w.save_results(world_name,result_filename,kq,kQ,kZ,kY,theta_fiedler,theta_CRB,seed_,runtime)
    return

    # with open(world_name+'/results.csv', 'a', newline='') as csvfile:
    #     results_writer = csv.writer(csvfile)
    #     results_writer.writerow([kq,kQ,kZ,kY,w.t,steps_taken,np.count_nonzero(w.base.grid),endtime-starttime,seed_])

if __name__ == "__main__":

    world_name = sys.argv[1]
    config_filename = sys.argv[2]
    result_filename = sys.argv[3]
    with open(world_name+'/'+config_filename) as json_file:
        config = json.load(json_file)

    progress = 1
    for kq in [0,0.1,0.01,0.001,0.5,0.0001,0.2,1]:
        for kQ in [0,0.1,0.01,0.001,0.5,0.0001,0.2,1]:
            for kZ in [0,0.1,0.01,0.001,0.5,0.0001,0.2,1]:
                for kY in [1]:
                    # kq, kQ, kZ, kY = -1, -1, -1, -1
                    # kq, kQ, kZ, kY = 0, 0, 0, 1
                    # kq, kQ, kZ, kY = 0.5, 0, 0, 1
                    # kq, kQ, kZ, kY = 0, 0, 0, 1
                    # kq, kQ, kZ, kY = 0.001, 0.001, 0.01, 1
                    # kq, kQ, kZ, kY = 1, 0.001, 0.001, 1
                    kq, kQ, kZ, kY = 1, 0, 0.001, 1
                    processes = []
                    # for seed_ in [0,1,2,3,4,5,6,7,8,9]:
                    for seed_ in [0]:
                        print("\n",progress,"out of",100)
                        progress += 1

                        if world_name == "nothing_world":
                            w = World(15,15,[7,7])
                            w.obstacles = []
                            w.targets = []
                            w.obstacles.append([7,7])
                            r1 = Robot(7,6,w)
                            r2 = Robot(7,8,w)
                            r3 = Robot(8,7,w)
                            r4 = Robot(8,8,w)

                        elif world_name == "wall_world":
                            w = World(15,15,[7,7])
                            w.obstacles = [[6,5],[7,5],[8,5],[9,5],[6,11],[7,11],[8,11],[9,11]]
                            w.targets = []
                            w.obstacles.append([7,7])
                            r1 = Robot(7,6,w)
                            r2 = Robot(7,8,w)
                            r3 = Robot(8,7,w)
                            r4 = Robot(8,8,w)

                        elif world_name == "obstacle_world":
                            w = World(15,15,[7,7])
                            w.obstacles = [[1,7],[3,3],[3,4],[8,2],[8,3],[12,5],[14,11],[13,11],[2,14],[9,11],[9,3]]
                            w.targets = []#[[2,4],[11,11],[1,14],[6,7],[9,2]]
                            w.obstacles.append([7,7])
                            r1 = Robot(7,6,w)
                            r2 = Robot(7,8,w)
                            r3 = Robot(8,7,w)
                            r4 = Robot(8,8,w)

                        elif world_name == "lavatube_world":
                            w = World(15,15,[7,7])
                            w.obstacles = [[1,1],[1,2],[1,3],[1,4],[2,4],[2,5],[3,5],[3,6],[4,6],[4,7],[5,7],[4,8],[3,8],[3,9],[2,9],[2,10],[1,10],[1,11],[1,12],[1,13],[2,13],[3,13],[4,13],[5,12],[5,13],[6,11],[6,12],[7,10],[7,11],[8,9],[8,10],[9,9],[10,9],[11,9],[12,9],[13,9],[14,9],[14,8],[14,7],[14,6],[14,5],[13,5],[12,5],[11,5],[10,5],[9,5],[8,5],[8,4],[7,4],[7,3],[6,3],[6,2],[5,2],[5,1],[4,1],[3,1],[2,1]]
                            w.targets = []
                            w.obstacles.append([7,7])
                            r1 = Robot(7,6,w)
                            r2 = Robot(7,8,w)
                            r3 = Robot(8,7,w)
                            r4 = Robot(8,8,w)

                        elif world_name == "baby_world":
                            w = World(5,5,[2,2])
                            w.obstacles = []
                            w.targets = []
                            w.obstacles.append([2,2])
                            r1 = Robot(2,1,w)
                            r2 = Robot(2,3,w)
                            r3 = Robot(3,1,w)
                            r4 = Robot(3,2,w)

                        elif world_name == "giant_world":
                            w = World(30,30,[15,15])
                            w.obstacles = [[22, 22], [11, 26], [28, 4], [11, 17], [0, 14], [11, 23], [23, 9], [13, 9], [14, 0], [22, 21], [19, 11], [4, 17], [12, 23], [20, 10], [11, 2], [24, 17], [17, 11], [26, 16], [23, 26], [14, 22]]
                            # np.random.seed(0)
                            # w.obstacles = np.random.randint(0,30,(75,2)).tolist()
                            w.targets = []
                            w.obstacles.append([15,15])
                            # r1 = Robot(14,14,w)
                            # r2 = Robot(14,16,w)
                            # r3 = Robot(16,14,w)
                            # r4 = Robot(16,16,w)
                            # r5 = Robot(13,15,w)
                            # r6 = Robot(17,15,w)
                            r1 = Robot(14,16,w)
                            r2 = Robot(15,16,w)
                            r3 = Robot(16,16,w)
                            r4 = Robot(14,17,w)
                            r5 = Robot(15,17,w)
                            r6 = Robot(16,17,w)

                            # r7 = Robot(14,14,w)
                            # r8 = Robot(15,14,w)
                            # r9 = Robot(16,14,w)
                            # r10 = Robot(14,13,w)
                            # r11 = Robot(15,13,w)
                            # r12 = Robot(16,13,w)

                        # runtime = lyapunov_exploration(w, kq, kQ, kZ, kY, b=config["B"], theta_fiedler=config["THETA_FIEDLER"], theta_CRB=config["THETA_CRB"], show=False, shuffle=config["SHUFFLE"], T=config["T"],seed_=seed_)
                        # p = multiprocessing.Process(target=lyapunov_exploration, args=(w, kq, kQ, kZ, kY, b=config["B"], theta_fiedler=config["THETA_FIEDLER"], theta_CRB=config["THETA_CRB"], show=False, shuffle=config["SHUFFLE"], T=config["T"], seed=seed_))
                        p = multiprocessing.Process(target=lyapunov_exploration, args=(w,kq,kQ,kZ,kY,config["B"],config["THETA_FIEDLER"],config["THETA_CRB"],False,config["SHUFFLE"],config["T"],seed_))
                        processes.append(p)
                        p.start()
                    for process in processes:
                        process.join()
                    assert False

    for kq, kQ, kZ, kY in [[-1,-1,-1,-1],[0,0,0,0]]:
        processes = []
        for seed_ in [0,1,2,3,4,5,6,7,8,9]:
            if world_name == "obstacle_world":
                w = World(15,15,[7,7])
                w.obstacles = [[1,7],[3,3],[3,4],[8,2],[8,3],[12,5],[14,11],[13,11],[2,14],[9,11],[9,3]]
                w.targets = []#[[2,4],[11,11],[1,14],[6,7],[9,2]]
                w.obstacles.append([7,7])
                r1 = Robot(7,6,w)
                r2 = Robot(7,8,w)
                r3 = Robot(8,7,w)
                r4 = Robot(8,8,w)

            elif world_name == "giant_world":
                w = World(30,30,[15,15])
                w.obstacles = [[22, 22], [11, 26], [28, 4], [11, 17], [0, 14], [11, 23], [23, 9], [13, 9], [14, 0], [22, 21], [19, 11], [4, 17], [12, 23], [20, 10], [11, 2], [24, 17], [17, 11], [26, 16], [23, 26], [14, 22]]
                w.targets = []
                w.obstacles.append([15,15])
                r1 = Robot(14,14,w)
                r2 = Robot(14,16,w)
                r3 = Robot(16,14,w)
                r4 = Robot(16,16,w)
                r5 = Robot(13,15,w)
                r6 = Robot(17,15,w)

            p = multiprocessing.Process(target=lyapunov_exploration, args=(w,kq,kQ,kZ,kY,config["B"],config["THETA_FIEDLER"],config["THETA_CRB"],False,config["SHUFFLE"],config["T"],seed_))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
