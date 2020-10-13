#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
from itertools import permutations
from scipy.special import comb
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})
import time
import sys
import json
import csv
from datetime import datetime
from grid_utils import *

world_name = sys.argv[1]
config_filename = sys.argv[2]
result_filename = sys.argv[3]
with open(world_name+'/'+config_filename) as json_file:
    config = json.load(json_file)

TRANSMIT_RADIUS_comm = config["TRANSMIT_RADIUS_comm"] #4
TRANSMIT_RADIUS_uwb = config["TRANSMIT_RADIUS_uwb"] #4
SENSING_RADIUS = config["SENSING_RADIUS"] #1.5
TIMEOUT = config["TIMEOUT"] #5
LIFETIME = config["LIFETIME"] #1000

MTTF = config["MTTF"] #50
MTTF_preventable = config["MTTF_preventable"] #25

UNKNOWN = config["UNKNOWN"] #0.0
FREE = config["FREE"] #1.0
OCCUPIED = config["OCCUPIED"] #-1.0
TARGET = config["TARGET"] #1.5
ROBOT = config["ROBOT"] #2.0
DEAD_ROBOT = config["DEAD_ROBOT"] #-0.5
LOST_ROBOT = config["LOST_ROBOT"] #0.5

class World():
    def __init__(self, rows, columns, basestate):
        self.shape = (rows,columns)
        self.robots = []
        self.true_states = {}
        self.base = Base(self, basestate[0], basestate[1])
        self.t = 0
        self.obstacles = []
        self.targets = []

    def generate_failure_times(self):
        # TODO
        n = len(self.robots)
        self.failure_times = np.random.exponential(MTTF,n)
        if MTTF_preventable == False:
            self.preventable_failure_times = [LIFETIME]*n
        else:
            self.preventable_failure_times = np.random.exponential(MTTF_preventable,n)
        print(self.failure_times,self.preventable_failure_times)

    def increment_time(self):
        for r in self.get_robots():
            # robot failure is possible
            if self.t == int(self.failure_times[r.id]) or r.conn_t == int(self.preventable_failure_times[r.id]):
                print("Robot",r.id,"failed at time",self.t)
                self.obstacles.append(r.state.copy())
                r.alive = False
            # timeouts
            timed_out = []
            for other_robot in r.known_states:
                t = r.known_states[other_robot][1]
                if t <= (self.t - TIMEOUT):
                    timed_out.append(other_robot)
            for gone in timed_out:
                r.known_states.pop(gone)
        # mission failure is possible
        not_lost = []
        for r in self.get_robots():
            if not r.lost:
                not_lost.append(r)
        if len(not_lost) == 0:
            return False
        self.t += 1 # increment simulation time
        for r in self.get_robots():
            if not r.conn:
                r.conn_t += 1 # increment failure clock
        return True

    def get_robots(self):
        alive_robots = []
        for r in self.robots:
            if r.alive:
                alive_robots.append(r)
        return alive_robots

    def add_robot(self,robot):
        self.robots.append(robot)
        self.true_states[robot] = [robot.state,self.t]

    def show_world(self,printout=True,plot=False,save=False):
        grid = self.base.grid.copy()
        for obstacle in self.obstacles:
            grid[obstacle[0],obstacle[1]] = OCCUPIED
        for target in self.base.known_targets:
            grid[target[0],target[1]] = TARGET
        for r in self.robots:
            if r.alive:
                if r.lost:
                    grid[r.state[0],r.state[1]] = LOST_ROBOT
                else:
                    grid[r.state[0],r.state[1]] = ROBOT
            else:
                grid[r.state[0],r.state[1]] = DEAD_ROBOT
        if printout:
            print(grid)
        if plot:
            plt.imshow(grid)
            plt.scatter(self.base.state[0],self.base.state[1],marker='x',c="red")
            plt.draw()
            plt.pause(0.01)
        if save:
            plt.imshow(grid)
            plt.scatter(self.base.state[0],self.base.state[1],marker='x',c="red")
            # try:
            #     plt.title(self.metrics["fiedler"][-1])
            # except:
            #     pass
            plt.savefig("./tmp/"+datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")+".png")

    def make_metrics(self):
        self.metrics = {}
        self.metrics["time"] = []
        self.metrics["coverage"] = []
        self.metrics["fiedler"] = []
        for r in self.robots:
            self.metrics[str(r.id)+"_kprob"] = []
            self.metrics[str(r.id)+"_CRB"] = []
            self.metrics[str(r.id)+"_queue"] = []
        self.metrics["true_states"] = []

    def log_metrics(self):
        self.metrics["time"].append(self.t)
        self.metrics["coverage"].append(np.count_nonzero(self.base.grid))
        self.metrics["fiedler"].append(calc_fiedler([x[0] for x in list(self.true_states.values())]+[self.base.state],TRANSMIT_RADIUS_comm,len(self.true_states),self.base.grid))
        states = [self.base.state] + [r.state for r in self.get_robots()]
        for r in self.robots:
            middle_states = []
            for other_r in self.get_robots():
                if other_r.id != r.id:
                    middle_states.append(other_r.state)
            if r.alive:
                # self.metrics[str(r.id)+"_kprob"].append(calc_multihop_link_prob(r.state,self.base.state,middle_states,TRANSMIT_RADIUS_comm,r.grid))
                # self.metrics[str(r.id)+"_kprob"].append(calc_khop_conn(r.state,self.base.state,middle_states,TRANSMIT_RADIUS_comm,r.grid))
                self.metrics[str(r.id)+"_CRB"].append(calc_CRB(r.state,states,TRANSMIT_RADIUS_uwb,len(states)-1,r.grid))
                self.metrics[str(r.id)+"_queue"].append(r.untransferred_data)
            else:
                self.metrics[str(r.id)+"_kprob"].append(np.nan)
                self.metrics[str(r.id)+"_CRB"].append(np.nan)
                self.metrics[str(r.id)+"_queue"].append(np.nan)
        self.metrics["true_states"].append([x[0] for x in list(self.true_states.values())])

    def save_results(self,world_name,result_filename,kq,kQ,kZ,kY,theta_fiedler,theta_CRB,seed_,runtime):
        with open(world_name+'/'+result_filename, 'a', newline='') as csvfile:
            results_writer = csv.writer(csvfile)
            coverage = self.metrics["coverage"][-1]
            time = self.metrics["time"][-1]
            time_connected = np.sum(np.array(self.metrics["fiedler"]) >= theta_fiedler)
            average_fiedler = np.mean(np.array(self.metrics["fiedler"]))
            time_localized = 0
            average_CRB = []
            max_queue = 0
            for r in self.robots:
                time_loc = np.array(self.metrics[str(r.id)+"_CRB"])
                time_loc = time_loc[~np.isnan(time_loc)]
                time_localized += np.sum(time_loc <= theta_CRB)
                average_CRB.append(np.mean(time_loc))
                if np.max(self.metrics[str(r.id)+"_queue"]) > max_queue:
                    max_queue = np.max(self.metrics[str(r.id)+"_queue"])
            time_localized = time_localized/len(self.robots)
            average_CRB = np.mean(average_CRB)
            results_writer.writerow([kq,kQ,kZ,kY,coverage,time,time_connected,average_fiedler,time_localized,average_CRB,max_queue,runtime,seed_,repr(self.metrics["coverage"])])

    # def plot_metrics(self,kq,kQ,kZ,kY,seed_,show=True,save=False):
    #     with open(world_name+"/plots/"+str(kq)+"_"+str(kQ)+"_"+str(kZ)+"_"+str(kY)+"_"+str(seed_)+".json", 'w') as fp:
    #         json.dump(self.metrics, fp, indent=2)
    #
    #     if show or save:
    #         fig1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
    #
    #         ax1.plot(self.metrics["time"],self.metrics["coverage"])
    #         ax1.set_title("Map size at data sink")
    #         # ax1.set_xlabel("Time")
    #         # ax1.set_ylabel("Map size (cells)")
    #
    #         ax2.plot(self.metrics["time"],self.metrics["fiedler"])
    #         ax2.set_title("Algebraic Connectivity (Fiedler Value)")
    #         # ax2.set_xlabel("Time")
    #         # ax2.set_ylabel("Fiedler value for graph including data sink")
    #
    #         for r in self.robots:
    #             ax3.plot(self.metrics["time"],self.metrics[str(r.id)+"_kprob"],label=str(r.id))
    #         ax3.set_title("Network Reliability")
    #         # ax3.set_xlabel("Time")
    #         # ax3.set_ylabel("K-hop connectivity")
    #
    #         for r in self.robots:
    #             ax4.plot(self.metrics["time"],self.metrics[str(r.id)+"_CRB"],label=str(r.id))
    #         ax4.set_title("Cramer Rao Bound")
    #         # ax4.set_xlabel("Time")
    #         # ax4.set_ylabel("Cramer Rao Bound")
    #
    #         for r in self.robots:
    #             ax5.plot(self.metrics["time"],self.metrics[str(r.id)+"_queue"],label=str(r.id))
    #         ax5.set_title("Queue size")
    #         ax5.set_xlabel("Time")
    #         # ax5.set_ylabel("Queue Size")
    #
    #         plt.tight_layout(pad=1.0)
    #         if save:
    #             plt.savefig(world_name+"/plots/"+str(kq)+"_"+str(kQ)+"_"+str(kZ)+"_"+str(kY)+".png")
    #         if show:
    #             plt.show()

class Robot():
    def __init__(self,x,y,world):
        self.state = [x,y]
        self.alive = True
        self.lost = False
        self.stuck = False
        self.conn = True
        self.conn_t = 0

        self.world = world
        world.add_robot(self)
        self.id = len(world.robots)-1
        self.grid = np.full(self.world.shape,UNKNOWN)
        self.grid[self.world.base.state[0],self.world.base.state[1]] = -1
        self.known_states = {self:[self.state,self.world.t]}

        # self.calculate_uncertainty()
        self.untransferred_data = 0 # the literal queue

        self.known_targets = []

        # assume we start connected and in a localizable state
        self.Q = 0 # these are the lyapunov queues
        self.Z = 0 # these are the lyapunov queues

    # not used
    def prob_successful_sensing(self):
        return 1

    def rate_frontier(self,f):
        rating = 0 # up to 9
        neighbors = adjacent_cells(f[0],f[1])
        for n in neighbors:
            if n[0] >= 0 and n[0] < self.grid.shape[0] and n[1] >= 0 and n[1] < self.grid.shape[1]:
                if self.grid[n[0],n[1]] == UNKNOWN:
                    rating += 1
        return rating

    def find_closest_frontier(self,search_stop=10,space=None):
        ''' find closest space with new data '''
        best_space = None
        best_dist = search_stop
        best_info = 0
        if space is None:
            space = self.state
        Q = [[space[0],space[1]]]
        visited = []

        while Q:
            x,y = Q.pop(0)
            if np.linalg.norm(np.subtract(space,[x,y])) > best_dist:
                break

            if self.grid[x,y] == FREE:
                f = [x,y]
                I = self.rate_frontier(f)
                if I > best_info:
                    best_space = f
                    best_dist = np.linalg.norm(np.subtract(space,f))
                    best_info = I

            visited.append([x,y])
            c_ns = cell_neighbors(x,y)
            for c_n in c_ns:
                if c_n[0] >= 0 and c_n[0]<self.grid.shape[0] and c_n[1] >= 0 and c_n[1] < self.grid.shape[1] and c_n not in visited and c_n not in Q:
                    Q.append(c_n)
            Q.sort(key=lambda x: np.linalg.norm(np.subtract(space,x)))

        if best_space:
            return best_space
        return self.world.base.state

    def get_constrained_space(self):
        state_space = get_state_space(self.state[0],self.state[1])
        constrained_space = []
        for space in state_space:
            # check space is in the map
            if space[0] >= 0 and space[0]<self.grid.shape[0] and space[1] >= 0 and space[1] < self.grid.shape[1]:
                # check not an obstacle:
                if self.grid[space[0],space[1]] != OCCUPIED:
                    # check collision avoidance
                    is_free = True
                    for r in self.known_states.keys():
                        if r != self:
                            if self.known_states[r][0] == space:
                                is_free = False
                    if is_free:
                        constrained_space.append(space)
        return constrained_space

    def constrained_optimize(self,theta_fiedler,theta_CRB,shuffle=False):
        cspace = self.get_constrained_space()
        if shuffle:
            random.shuffle(cspace)
        best_score = np.inf
        best = self.state
        search_stop = 100
        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            if r != self:
                states.append(self.known_states[r][0])

        for space in cspace:
            fiedler = calc_fiedler(states+[space,base],TRANSMIT_RADIUS_comm,len(states)+1,self.grid)
            # CRB = calc_CRB(space,states+[space],TRANSMIT_RADIUS_uwb,len(states),self.grid)
            CRB = calc_CRB(space,states+[space,base],TRANSMIT_RADIUS_uwb,len(states)+1,self.grid)

            if self.stuck and self.untransferred_data:
                if fiedler >= theta_fiedler and CRB <= theta_CRB:
                    _, score = path_plan(space,base,self.grid)
                else:
                    score = np.inf
            if self.stuck and self.untransferred_data == 0:
                self.stuck = False

            if not self.stuck:
                if fiedler >= theta_fiedler and CRB <= theta_CRB:
                    f = self.find_closest_frontier(search_stop,space)
                    _, dist_f = path_plan(space,f,self.grid)
                    score = dist_f
                else:
                    score = np.inf

            if score < best_score:
                best_score = score
                best = space

        if best == self.state:
            # self.stuck = True
            pass
        return best, best_score

    def lyapunov_optimize(self,kq,kQ,kZ,kY,b,theta_fiedler,theta_CRB,shuffle=False):
        cspace = self.get_constrained_space()
        if shuffle:
            # shuffle the cspace to add some randomness
            random.shuffle(cspace)

        best_score = np.inf
        search_stop = 100

        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            if r != self:
                states.append(self.known_states[r][0])

        # scores = []
        for space in cspace:
            f = self.find_closest_frontier(search_stop,space)
            _, dist_f = path_plan(space,f,self.grid)
            I = self.rate_frontier(space)
            if len(self.world.robots) <= 4:
                p_conn = calc_multihop_link_prob(space,base,states,TRANSMIT_RADIUS_comm,self.grid)
            else:
                p_conn = calc_khop_conn(space,base,states,TRANSMIT_RADIUS_comm,self.grid)
            fiedler = calc_fiedler(states+[space,base],TRANSMIT_RADIUS_comm,len(states)+1,self.grid)
            # CRB = calc_CRB(space,states+[space],TRANSMIT_RADIUS_uwb,len(states),self.grid)
            CRB = calc_CRB(space,states+[space,base],TRANSMIT_RADIUS_uwb,len(states)+1,self.grid)

            # if self.stuck and self.untransferred_data:
            #     _, score = path_plan(space,base,self.grid)
            # if self.stuck and self.untransferred_data == 0:
            #     self.stuck = False

            if self.stuck:
                if calc_link_prob(space,base,TRANSMIT_RADIUS_comm,self.grid) > 0.5:
                    self.stuck = False
                else:
                    _, score = path_plan(space,base,self.grid)

            '''Here is the DPP objective'''
            if not self.stuck:
                score = kY*dist_f + kq*self.untransferred_data*I*(1-p_conn) - kq*self.untransferred_data*b*p_conn + kQ*self.Q*(theta_fiedler-fiedler) + kZ*self.Z*(CRB-theta_CRB)
                # scores.append(score)

            if score < best_score:
                best_score = score
                best = space

        # print(scores)
        if best == self.state:
            self.stuck = True

        # update lyapunov queues
        self.Q = max(self.Q + (theta_fiedler-fiedler), 0)
        self.Z = max(self.Z + (CRB-theta_CRB), 0)
        return best, best_score

    def random_step(self):
        cspace = self.get_constrained_space()
        best = random.choice(cspace)
        return best, None

    def move(self,space):
        if space and self.alive:
            if space in self.world.obstacles:
                self.alive = False
            for r in self.world.true_states:
                if self.world.true_states[r][0] == space:
                    return
                    # print("uh oh, collision at state",space)
                    # r.alive = False
                    # self.alive = False
                    # self.world.obstacles += [r.state,self.state]
                    # return
            self.state[0] = space[0]
            self.state[1] = space[1]
            self.world.true_states[self] = [self.state,self.world.t]
            self.known_states[self] = [self.state,self.world.t]
            other_states = []
            for r in self.known_states.keys():
                if r != self:
                    other_states.append(self.known_states[r][0])

    def calculate_uncertainty(self):
        states = []
        for r in self.world.get_robots():
            states.append(r.state)
        self.CRB = calc_CRB(self.state,states,TRANSMIT_RADIUS_uwb,len(states),self.grid)
        # if self.CRB > 500:
        #     print("robot is lost at time",self.world.t,"with CRB",self.CRB)
        #     self.lost = True

    def explore(self,space):
        if not self.lost:
            Q = [[space[0],space[1]]]
            visited = []
            while Q:
                x,y = Q.pop(0)
                ### if outside sensing radius, stop
                if np.linalg.norm(np.subtract(space,[x,y])) > SENSING_RADIUS:
                    break
                ### if within sensing radius and unknown
                if self.grid[x,y] == UNKNOWN:
                    self.untransferred_data +=1
                    if [x,y] in self.world.obstacles:
                        self.grid[x,y] = OCCUPIED
                    elif [x,y] in self.world.targets:
                        self.known_targets.append([x,y])
                        self.grid[x,y] = FREE
                        # TODO how much extra are science targets worth?
                        self.untransferred_data += 1
                    else:
                        self.grid[x,y] = FREE
                elif [x,y] in self.world.obstacles:
                    self.grid[x,y] = OCCUPIED
                ### add neighbors to the queue
                visited.append([x,y])
                c_ns = cell_neighbors(x,y)
                for c_n in c_ns:
                    if c_n[0] >= 0 and c_n[0]<self.grid.shape[0] and c_n[1] >= 0 and c_n[1] < self.grid.shape[1] and c_n not in visited and c_n not in Q:
                        Q.append(c_n)
                Q.sort(key=lambda x: np.linalg.norm(np.subtract(space,x)))

    def share(self):
        k = len(self.world.get_robots())
        middle_states = []
        for r in self.world.get_robots():
            if r.id != self.state:
                middle_states.append(r.state)

        if len(self.world.robots) <= 4:
            link = sim_multihop_link(self.state, self.world.base.state, middle_states, TRANSMIT_RADIUS_comm, self.world.obstacles)
        else:
            link = sim_khop_conn(self.state, self.world.base.state, middle_states, TRANSMIT_RADIUS_comm, self.world.obstacles)
        if link:
            self.conn = True
            self.untransferred_data = 0
            self.world.base.grid += self.grid
            self.world.base.grid = np.clip(self.world.base.grid,OCCUPIED,FREE)
            self.world.base.known_states[self] = [self.state,self.world.t]
            for target in self.known_targets:
                if target not in self.world.base.known_targets:
                    self.world.base.known_targets.append(target)
        else:
            self.conn = False

        for r in self.world.get_robots():
            if len(self.world.robots) <= 4:
                link = sim_multihop_link(self.state, r.state, middle_states, TRANSMIT_RADIUS_comm, self.world.obstacles)
            else:
                link = sim_khop_conn(self.state, r.state, middle_states, TRANSMIT_RADIUS_comm, self.world.obstacles)
            if link:
                r.grid += self.grid
                r.grid = np.clip(r.grid,OCCUPIED,FREE)
                r.known_states[self] = [self.state,self.world.t]
                for target in self.known_targets:
                    if target not in r.known_targets:
                        r.known_targets.append(target)


class Base():
    def __init__(self,world,x,y):
        self.state = [x,y]
        self.grid = np.full(world.shape,UNKNOWN)
        self.grid[x,y] = -1
        self.world = world
        self.known_states = {}
        self.known_targets = []
