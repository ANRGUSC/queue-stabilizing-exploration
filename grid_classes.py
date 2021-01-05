#!/usr/bin/env python
# coding: utf-8
import os
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
result_filename = config_filename[:-5]+".csv"
# result_filename = sys.argv[3]
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
# not used
LOST_ROBOT = config["LOST_ROBOT"] #0.5

MAX_Q = config["MAX_Q"] #300
THETA_DELAY = config["THETA_DELAY"] #0.1
B = config["B"] #225
THETA_CRB = config["THETA_CRB"] #20
THETA_FIEDLER = config["THETA_FIEDLER"] #2
SHUFFLE = config["SHUFFLE"] #true
T = config["T"] #1000

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
        n = len(self.robots)
        if MTTF == False:
            self.failure_times = [LIFETIME]*n
        else:
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
        self.t += 1 # increment simulation time
        for r in self.get_robots():
            if not r.conn:
                r.conn_t += 1 # increment failure clock
        return len(self.get_robots())>0

    def get_robots(self):
        alive_robots = []
        for r in self.robots:
            if r.alive:
                alive_robots.append(r)
        return alive_robots

    def add_robot(self,robot):
        self.robots.append(robot)
        self.true_states[robot] = [robot.state,self.t]

    def show_world(self,gains,printout=True,plot=False,save=False):
        grid = self.base.grid.copy()
        for obstacle in self.obstacles:
            grid[obstacle[0],obstacle[1]] = OCCUPIED
        for target in self.base.known_targets:
            grid[target[0],target[1]] = TARGET
        for r in self.robots:
            if r.alive:
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
            result_foldername = result_filename[:-4]+''.join('_'+str(k) for k in gains)
            if not os.path.isdir(world_name+"/"+result_foldername):
                os.makedirs(world_name+"/"+result_foldername)
            plt.savefig(world_name+"/"+result_foldername+"/"+datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")+".png")

    def make_metrics(self):
        self.metrics = {}
        self.metrics["time"] = []
        self.metrics["coverage"] = []
        self.metrics["fiedler"] = []
        for r in self.robots:
            self.metrics[str(r.id)+"_CRB"] = []
            self.metrics[str(r.id)+"_queue"] = []

    def log_metrics(self):
        self.metrics["time"].append(self.t)
        # coverage known at base
        self.metrics["coverage"].append(np.count_nonzero(self.base.grid))
        # connectivity of all robots and base
        states = [self.base.state] + [r.state for r in self.get_robots()]
        self.metrics["fiedler"].append(calc_fiedler(states,TRANSMIT_RADIUS_comm,len(self.true_states),self.base.grid))
        for r in self.robots:
            middle_states = []
            for other_r in self.get_robots():
                if other_r.id != r.id:
                    middle_states.append(other_r.state)
            if r.alive:
                self.metrics[str(r.id)+"_CRB"].append(calc_CRB(r.state,states,TRANSMIT_RADIUS_uwb,len(states)-1,r.grid))
                self.metrics[str(r.id)+"_queue"].append(r.untransferred_data)
            else:
                self.metrics[str(r.id)+"_CRB"].append(np.nan)
                self.metrics[str(r.id)+"_queue"].append(np.nan)

    def save_results(self,kq,kQ,kZ,kY,seed_,runtime):
        if not os.path.exists(world_name+'/'+result_filename):
            with open(world_name+'/'+result_filename, 'a', newline='') as csvfile:
                results_writer = csv.writer(csvfile,quotechar = "'")
                results_writer.writerow(["\""+x+"\"" for x in ["kq","kQ","kZ","kY","coverage","time","time_connected","average_fiedler","time_localized","average_CRB","max_queue","runtime","seed","coverage_array"]])
        with open(world_name+'/'+result_filename, 'a', newline='') as csvfile:
            results_writer = csv.writer(csvfile)
            coverage = self.metrics["coverage"][-1]
            time = self.metrics["time"][-1]
            time_connected = np.sum(np.array(self.metrics["fiedler"]) >= THETA_FIEDLER)
            average_fiedler = np.mean(np.array(self.metrics["fiedler"]))
            time_localized = 0
            average_CRB = []
            max_queue = 0
            for r in self.robots:
                time_loc = np.array(self.metrics[str(r.id)+"_CRB"])
                time_loc = time_loc[~np.isnan(time_loc)]
                time_localized += np.sum(time_loc <= THETA_CRB)
                average_CRB.append(np.mean(time_loc))
                if np.max(self.metrics[str(r.id)+"_queue"]) > max_queue:
                    max_queue = np.max(self.metrics[str(r.id)+"_queue"])
            # average over number of robots
            time_localized = time_localized/len(self.robots)
            average_CRB = np.mean(average_CRB)
            results_writer.writerow([kq,kQ,kZ,kY,coverage,time,time_connected,average_fiedler,time_localized,average_CRB,max_queue,runtime,seed_,repr(self.metrics["coverage"])])

class Robot():
    def __init__(self,x,y,world):
        self.state = [x,y]
        self.alive = True
        self.stuck = False
        self.conn = True
        self.conn_t = 0

        self.world = world
        world.add_robot(self)
        self.id = len(world.robots)-1
        self.grid = np.full(self.world.shape,UNKNOWN)
        self.grid[self.world.base.state[0],self.world.base.state[1]] = -1
        self.known_states = {}

        # self.calculate_uncertainty()
        self.untransferred_data = 0 # the literal queue
        self.q = 0 # the delay queue

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

    def find_closest_frontier(self,search_stop=100,space=None):
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
                        if self.known_states[r][0] == space:
                            is_free = False
                    if is_free:
                        constrained_space.append(space)
        return constrained_space

    def time_preference_optimize(self,ratio):
        cspace = self.get_constrained_space()
        current, others = cspace[0], cspace[1:]
        if SHUFFLE:
            random.shuffle(others)
        cspace = [current] + others
        best_score = np.inf
        best = self.state
        search_stop = 100
        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            states.append(self.known_states[r][0])

        if (np.count_nonzero(self.grid)-self.untransferred_data)/np.count_nonzero(self.grid) < ratio:
            # print(np.count_nonzero(self.grid), self.untransferred_data, "return")
            for space in cspace:
                # TODO pathplan could just return the next point as the best
                _, score = path_plan(space,base,self.grid)
                if score <= best_score:
                    best = space
                    best_score = score

        else:
            # print("explore")
            f = self.find_closest_frontier(search_stop,current)
            for space in cspace:
                # f = self.find_closest_frontier(search_stop,space)
                _, dist_f = path_plan(space,f,self.grid)
                score = dist_f
                if score <= best_score:
                    best = space
                    best_score = score

        return best, best_score

    def constrained_optimize(self):
        cspace = self.get_constrained_space()
        current, others = cspace[0], cspace[1:]
        if SHUFFLE:
            random.shuffle(others)
        cspace = [current] + others
        best_score = np.inf
        best = self.state
        search_stop = 100
        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            states.append(self.known_states[r][0])

        f = self.find_closest_frontier(search_stop,current)
        for space in cspace:
            fiedler = calc_fiedler(states+[space,base],TRANSMIT_RADIUS_comm,len(states)+1,self.grid)
            CRB = calc_CRB(space,states+[space,base],TRANSMIT_RADIUS_uwb,len(states)+1,self.grid)

            if fiedler >= THETA_FIEDLER and CRB <= THETA_CRB:
                # f = self.find_closest_frontier(search_stop,space)
                _, dist_f = path_plan(space,f,self.grid)
                score = dist_f
            else:
                score = np.inf

            if score < best_score:
                best_score = score
                best = space

        return best, best_score

    def unconstrained_optimize(self):
        cspace = self.get_constrained_space()
        current, others = cspace[0], cspace[1:]
        if SHUFFLE:
            random.shuffle(others)
        cspace = [current] + others
        best_score = np.inf
        best = self.state
        search_stop = 100
        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            states.append(self.known_states[r][0])

        f = self.find_closest_frontier(search_stop,current)
        for space in cspace:
            # f = self.find_closest_frontier(search_stop,space)
            _, dist_f = path_plan(space,f,self.grid)
            score = dist_f

            if score <= best_score:
                best_score = score
                best = space

        return best, best_score

    def lyapunov_optimize(self,kq,kQ,kZ,kY):
        cspace = self.get_constrained_space()
        # keep the current cell first, so that if any other option is better or equal the robot will move
        current, others = cspace[0], cspace[1:]
        if SHUFFLE:
            random.shuffle(others)
        cspace = [current] + others

        best_score = np.inf
        best_fiedler = None
        best_CRB = None
        # for debugging
        min_dist_f = np.inf
        min_dist_f_p_conn = 0
        best_dist_f = np.inf
        best_p_conn = 0

        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            states.append(self.known_states[r][0])

        # TODO should this be the actual link rather than the expectation?
        if calc_link_prob(current, base, TRANSMIT_RADIUS_comm,self.grid) > 0.5:
            self.stuck = False

        search_stop = 100
        f = self.find_closest_frontier(search_stop,current)

        for space in cspace:
            fiedler = calc_fiedler(states+[space,base],TRANSMIT_RADIUS_comm,len(states)+1,self.grid)
            CRB = calc_CRB(space,states+[space,base],TRANSMIT_RADIUS_uwb,len(states)+1,self.grid)

            # score is dist to base until unstuck
            if self.stuck:
                _, score = path_plan(space,base,self.grid)
                # TODO consider breaking local optima towards frontier if q is empty

            else:
                # f = self.find_closest_frontier(search_stop,space)
                _, dist_f = path_plan(space,f,self.grid)
                I = self.rate_frontier(space)
                if len(self.world.robots) <= 4:
                    p_conn = calc_multihop_link_prob(space,base,states,TRANSMIT_RADIUS_comm,self.grid)
                else:
                    p_conn = calc_khop_conn(space,base,states,TRANSMIT_RADIUS_comm,self.grid)

                # for debugging
                if dist_f < min_dist_f:
                    min_dist_f = dist_f
                    min_dist_f_p_conn = p_conn

                '''Here is the DPP objective'''
                # literal queue dynamics
                # score = kY*dist_f + kq*self.untransferred_data*(I - B*p_conn) + kQ*self.Q*(THETA_FIEDLER-fiedler) + kZ*self.Z*(CRB-THETA_CRB)

                # delay queue grows with old q
                score = kY*dist_f + kq*self.q*(self.untransferred_data - THETA_DELAY*B*p_conn) + kQ*self.Q*(THETA_FIEDLER-fiedler) + kZ*self.Z*(CRB-THETA_CRB)

                # delay queue grows with new q
                # score = kY*dist_f + kq*self.q*(I - B*p_conn - THETA_DELAY*B*p_conn) + kQ*self.Q*(THETA_FIEDLER-fiedler) + kZ*self.Z*(CRB-THETA_CRB)

            if score <= best_score:
                best_score = score
                best = space
                best_fiedler = fiedler
                best_CRB = CRB
                if not self.stuck:
                    best_dist_f = dist_f
                    best_p_conn = p_conn

        # if not self.stuck:
        #     if best_dist_f > min_dist_f:
        #         # print("Sacrificing exploration. Queue size:",self.untransferred_data,"Delta:",best_p_conn-min_dist_f_p_conn)
        #         print(self.id,"Sacrificing exploration. Queue size:",self.untransferred_data,"Delay queue size:",self.q)

        if best == self.state: # current state is strictly better than other states
            # print("avoiding local optima")
            self.stuck = True

        # update lyapunov queues
        self.Q = max(self.Q + (THETA_FIEDLER-best_fiedler), 0)
        self.Z = max(self.Z + (best_CRB-THETA_CRB), 0)
        return best, best_score

    def multi_objective_optimize(self,kq,kQ,kZ,kY):
        # NEGATE kY because kY = -1 is the trigger to use this and not lyapunov_optimize
        kY = -kY
        assert kY > 0
        cspace = self.get_constrained_space()
        current, others = cspace[0], cspace[1:]
        if SHUFFLE:
            random.shuffle(others)
        cspace = [current] + others
        best_score = np.inf
        best = self.state
        search_stop = 100
        base = self.world.base.state
        states = []
        for r in self.known_states.keys():
            states.append(self.known_states[r][0])
        search_stop = 100
        f = self.find_closest_frontier(search_stop,current)
        for space in cspace:
            fiedler = calc_fiedler(states+[space,base],TRANSMIT_RADIUS_comm,len(states)+1,self.grid)
            CRB = calc_CRB(space,states+[space,base],TRANSMIT_RADIUS_uwb,len(states)+1,self.grid)
            # f = self.find_closest_frontier(search_stop,space)
            _, dist_f = path_plan(space,f,self.grid)
            if len(self.world.robots) <= 4:
                p_conn = calc_multihop_link_prob(space,base,states,TRANSMIT_RADIUS_comm,self.grid)
            else:
                p_conn = calc_khop_conn(space,base,states,TRANSMIT_RADIUS_comm,self.grid)
            # score = kY*dist_f + kq*p_conn + kQ*fiedler + kZ*CRB
            score = kY*dist_f - kq*p_conn - kQ*fiedler + kZ*CRB
            if score <= best_score:
                best_score = score
                best = space
        # TODO allow this to get stuck?
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
                    # print("would-be collision avoided by waiting...")
                    return
            self.state[0] = space[0]
            self.state[1] = space[1]
            self.world.true_states[self] = [self.state,self.world.t]

    def explore(self,space):
        self.u = self.untransferred_data
        if self.untransferred_data < MAX_Q:
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
        else:
            # print("Not exploring because Q_MAX has been exceeded, returning to base")
            self.stuck = True
            print("q_max was reached, not exploring")

    def share(self):
        k = len(self.world.get_robots())
        middle_states = [] # true states
        for r in self.world.get_robots():
            if r.id != self.id:
                middle_states.append(r.state)
        # try to connect to the base
        emptied_q = False
        self.conn = sim_khop_conn(self.state, self.world.base.state, middle_states, TRANSMIT_RADIUS_comm, self.world.obstacles)
        if self.conn:
            # TODO empty queue if link!
            self.untransferred_data = max(self.untransferred_data-B,0)
            self.q = max(self.q - THETA_DELAY*B, 0) + self.u
            emptied_q = True
            self.world.base.grid += self.grid
            self.world.base.grid = np.clip(self.world.base.grid,OCCUPIED,FREE)
            for target in self.known_targets:
                if target not in self.world.base.known_targets:
                    self.world.base.known_targets.append(target)
            self.grid += self.world.base.grid
            self.grid = np.clip(self.grid,OCCUPIED,FREE)
        # try to update any neighbors
        middle_states.append(self.world.base.state)
        for r in self.world.get_robots():
            if r.id != self.id:
                link = sim_khop_conn(self.state, r.state, middle_states, TRANSMIT_RADIUS_comm, self.world.obstacles)
                if link:
                    # TODO r can only accept my data if its queue is not full?
                    r.grid += self.grid
                    r.grid = np.clip(r.grid,OCCUPIED,FREE)
                    r.known_states[self] = [self.state,self.world.t]
                    for target in self.known_targets:
                        if target not in r.known_targets:
                            r.known_targets.append(target)
                    # TODO if this robot is closer, has space in its queue, and I wasn't able to empty my queue directly
                    if r.untransferred_data < MAX_Q:
                        if np.linalg.norm(np.subtract(r.state,self.world.base.state)) < np.linalg.norm(np.subtract(self.state,self.world.base.state)):
                            if not emptied_q:
                                r.untransferred_data += min(self.untransferred_data,B)
                                self.untransferred_data = max(self.untransferred_data-B,0)
                                # print("transfering my delay queue to my neighbor",self.id,r.id)
                                r.q += min(self.q,THETA_DELAY*B)
                                self.q = max(self.q - THETA_DELAY*B, 0) + self.u
                                emptied_q = True
        # if disconnected, delay queue grows
        if not emptied_q:
            self.q = self.q + self.u

class Base():
    def __init__(self,world,x,y):
        self.state = [x,y]
        self.grid = np.full(world.shape,UNKNOWN)
        self.grid[x,y] = -1
        self.world = world
        self.known_states = {}
        self.known_targets = []
