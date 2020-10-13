#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
from itertools import permutations, combinations
from scipy.special import comb
import matplotlib
import matplotlib.pyplot as plt
import time
import functools

def cell_neighbors(x,y):
    return [[x+1,y],[x,y+1],[x-1,y],[x,y-1]]

def adjacent_cells(x,y):
    return [[x+1,y],[x,y+1],[x-1,y],[x,y-1],[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1]]

def get_state_space(x,y):
    return [[x,y],[x+1,y],[x,y+1],[x-1,y],[x,y-1],[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1]]

def path_plan(s1,s2,grid):
    cells = {tuple(s1):[0,None]}
    visited = []
    Q = [s1]
    while Q:
        s = Q.pop(0)
        if s == s2:
            return s, cells[tuple(s)][0]
        if s not in visited:
            visited.append(s)
            cn = adjacent_cells(s[0],s[1])
            for n in cn:
                if tuple(n) in cells:
                    dist_to_n = cells[tuple(n)][0]
                else:
                    dist_to_n = np.inf
                dist_thru_s = cells[tuple(s)][0] + np.linalg.norm(np.subtract(s,n))
                if dist_thru_s < dist_to_n:
                    cells[tuple(n)] = [dist_thru_s,s]
                if n[0] >= 0 and n[0] < grid.shape[0] and n[1] >= 0 and n[1] < grid.shape[1]:
                    if grid[n[0],n[1]] == 1.0:
                        Q.append(n)
    return s1, np.linalg.norm(np.subtract(s1,s2))

def get_y(s1,s2,x):
    a,b = s1
    c,d = s2
    return (b-d)/(a-c)*(x-a)+b

def get_x(s1,s2,y):
    a,b = s1
    c,d = s2
    return (-a*d+a*y+b*c-c*y)/(b-d)

def calc_link_prob(s1, s2, d, grid):
    '''Predict link probability of success'''
    cells_between = []

    for x in list(range(s1[0],s2[0]))+list(range(s2[0],s1[0])):
        y = get_y(s1,s2,x+0.5)
        cells_between.append([x,int(np.floor(y+0.5))])
        cells_between.append([x+1,int(np.floor(y+0.5))])

    for y in list(range(s1[1],s2[1]))+list(range(s2[1],s1[1])):
        x = get_x(s1,s2,y+0.5)
        cells_between.append([int(np.floor(x+0.5)),y])
        cells_between.append([int(np.floor(x+0.5)),y+1])

    obstacles = []
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if grid[row][column] == -1 and [row,column] != s1 and [row,column] != s2:
                obstacles.append([row,column])

    for cell in cells_between:
        row,column = cell
        if cell in obstacles:
            return 0

    x = np.linalg.norm(np.subtract(s1,s2))
    p = 1/(1+np.e**(3*(x-d)))
    return p

def sim_link(s1,s2,d,obstacles):
    '''flip a coin to determine success of link'''
    cells_between = []

    for x in list(range(s1[0],s2[0]))+list(range(s2[0],s1[0])):
        y = get_y(s1,s2,x+0.5)
        cells_between.append([x,int(np.floor(y+0.5))])
        cells_between.append([x+1,int(np.floor(y+0.5))])

    for y in list(range(s1[1],s2[1]))+list(range(s2[1],s1[1])):
        x = get_x(s1,s2,y+0.5)
        cells_between.append([int(np.floor(x+0.5)),y])
        cells_between.append([int(np.floor(x+0.5)),y+1])

    obstructed = False
    for cell in cells_between:
        row,column = cell
        # grid[row,column] = 0.5
        # TODO
        if cell in obstacles and cell != s1 and cell != s2:
            # grid[row,column] = 0.25
            obstructed = True

    if not obstructed:
        x = np.linalg.norm(np.subtract(s1,s2))
        p = 1/(1+np.e**(3*(x-d)))
        odds = np.random.random()
        return odds < p
    else:
        return False

@functools.lru_cache(maxsize=32)
def paths(n_nodes):
    n_edges = int(n_nodes*(n_nodes-1)/2)
    paths = []
    Q = [(0,)]
    while Q:
        possible_path = np.zeros(n_edges)
        current_path = Q.pop()
        node = current_path[-1]
        for neighbor in range(n_nodes):
            if neighbor not in current_path:
                if neighbor == n_nodes-1:
                    path = current_path+(neighbor,)
                    path_arr = path2bin(path,n_nodes,n_edges)
                    paths.append(path_arr)
                else:
                    Q.append(current_path+(neighbor,))
    return np.array(paths)

@functools.lru_cache(maxsize=32)
def path2bin(path,n_nodes,n_edges):
    arr = np.zeros(n_edges)
    for i in range(len(path)-1):
        edge = [path[i],path[i+1]]
        edge.sort()
        index = int(1/2*edge[0]*(2*n_nodes - edge[0] - 1) + (edge[1]-edge[0]) - 1)
        arr[index] = 1
    return arr

def compare(P,S):
    ''' are they disjoint? is P contained in S? if not return variables in S but not in P'''
    if np.any(S[np.nonzero(P==0)]==1):
        return False
    return np.where(np.logical_and(np.isnan(P),S==1))[0].tolist()

def disjoint_set(S):
    ''' return all the disjoint products '''
    dis_set = np.array(S[0])
    # print("disjoint set:", dis_set)
    n = len(S)
    for k in range(1,n):
        PDk = [S[k]]
        for j in range(k):
            PDk_new = []
            for Pi in PDk:
                compare_output = compare(Pi,S[j])
                if compare_output == False:
                    # disjoint, keep
                    PDk_new.append(Pi)
                elif compare_output:
                    for idx_idx, idx in enumerate(compare_output):
                        newPi = Pi.copy()
                        for pre_idx in range(idx_idx):
                            newPi[compare_output[pre_idx]] = 1
                        newPi[idx] = 0
                        PDk_new.append(newPi)
            PDk = PDk_new
        dis_set = np.vstack([dis_set,np.array(PDk)])
    return dis_set

def fast_probn(edge_probs,n_nodes):
    S = paths(n_nodes)
    S[S==0] = np.nan
    products = disjoint_set(S)
    edge_probs_repeat = np.tile(edge_probs,(products.shape[0],1))
    p = np.product(np.where(products==1,edge_probs_repeat,1),1)
    q = np.product(np.where(products==0,1-edge_probs_repeat,1),1)
    return np.dot(p,q)

def calc_multihop_link_prob(state1,state2,middle_states,d,grid):
    ''' Calculate terminal reliability '''
    n_nodes = 2 + len(middle_states)
    n_edges = n_nodes*(n_nodes-1)/2
    edge_probs = []

    tx_states = [state1]+middle_states+[state2]

    while tx_states:
        s1 = tx_states.pop()
        for s2 in tx_states:
            edge_probs.append(calc_link_prob(s1,s2,d,grid))

    return fast_probn(np.array(edge_probs),n_nodes)
    # return probn(np.array(edge_probs),n_nodes)

def calc_khop_conn(state1,state2,middle_states,d,grid):
    '''predict roughly number of paths from state1 to state2'''
    '''use this as a heuristic when number of robots grows'''
    states = middle_states+[state1,state2]
    k = len(states)-1

    A = np.zeros([len(states),len(states)])
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if i < j:
                if s1 == state1:
                    index1 = i
                if s2 == state2:
                    index2 = j
                p = calc_link_prob(s1, s2, d, grid)
                A[i][j] = p
                A[j][i] = p
    np.fill_diagonal(A,0)

    s = np.identity(len(states))
    for power in range(1,k+1):
        s += np.linalg.matrix_power(A,power)

    return s[index1][index2]

def sim_khop_conn(state1,state2,middle_states,d,obstacles):
    '''' flip coins for each edge, then return if a simple path exists'''
    '''use this as a heuristic when number of robots grows'''
    states = middle_states+[state1,state2]
    k = len(states)-1

    A = np.zeros([len(states),len(states)])
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if i < j:
                if s1 == state1:
                    index1 = i
                if s2 == state2:
                    index2 = j
                l = sim_link(s1,s2,d,obstacles)
                A[i][j] = l
                A[j][i] = l
    np.fill_diagonal(A,0)

    s = np.identity(len(states))
    for power in range(1,k+1):
        s += np.linalg.matrix_power(A,power)

    return s[index1][index2]>0

def sim_multihop_link(state1,state2,middle_states,d,obstacles):
    ''' flip coins for each edge, then return if a simple path exists '''
    n_nodes = 2 + len(middle_states)
    n_edges = n_nodes*(n_nodes-1)/2
    edge_probs = []

    tx_states = [state1]+middle_states+[state2]

    while tx_states:
        s1 = tx_states.pop()
        for s2 in tx_states:
            edge_probs.append(sim_link(s1,s2,d,obstacles))

    return fast_probn(np.array(edge_probs),n_nodes)

def calc_fiedler(states,d,k,grid):
    A = np.zeros([len(states),len(states)])
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            A[i][j] = calc_link_prob(state1,state2,d,grid) > 0.5
    d = np.sum(A, axis=0)
    D = np.diag(d)
    L = D - A
    val, Vec = np.linalg.eigh(L)
    try:
        return np.sort(val)[1]
    except IndexError:
        return 0

# not used
def calc_rigidity_eigenvalue(states,d,k,grid):
    M = []
    for i, u in enumerate(states):
        for j, v in enumerate(states):
            if i < j:
                if calc_link_prob(u,v,d,grid) > 0.5:
                    row = [0]*(len(states)*2)
                    row[2*i], row[2*i+1] = u[0]-v[0],u[1]-v[1]
                    row[2*j], row[2*j+1] = v[0]-u[0],v[1]-u[1]
                    M.append(row)
    M = np.array(M)
    symmM = np.dot(M.T,M)
    val, Vec = np.linalg.eigh(symmM)
    lambda_4 = np.sort(val)[3]
    return np.round(lambda_4)

def calc_CRB(state1,states,d,k,grid):
    # TODO
    n = len(states)
    Vp = 3e8
    sigma_t = 6e-9
    gamma = 1/(Vp*sigma_t)**2
    s = 2

    FIM = np.zeros((2,2))
    for i in range(n):
        if states[i] != state1:
            if calc_link_prob(state1,states[i],d,grid) > 0.5:
                vec = np.subtract(state1,states[i])
                norm = np.linalg.norm(vec)
                FIM[0,0] += gamma*vec[0]**2/norm**s
                FIM[0,1] += gamma*vec[0]*vec[1]/norm**s
                FIM[1,0] += gamma*vec[0]*vec[1]/norm**s
                FIM[1,1] += gamma*vec[1]**2/norm**s

    try:
        CRB = np.trace(np.linalg.inv(FIM))
        if CRB >= 0 and CRB < 1000:
            return CRB
        else:
            return 1000
    except np.linalg.LinAlgError:
        return 1000
