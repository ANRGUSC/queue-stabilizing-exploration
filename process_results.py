import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

def compare_avg_coverage(world_name,result_filename,world_size):
    # fig1, (ax1) = plt.subplots(1,1,figsize=(5,5))
    fig1, (ax1) = plt.subplots(1,1,figsize=(8,4))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Map size at data sink (cells)")
    df = pd.read_csv(world_name+'/'+result_filename)
    # df = df[df["kY"]==-100]
    agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'time':['max']})
    agg_df.columns = ["max_time"]
    max_len = max(agg_df["max_time"])+2

    for i in range(len(agg_df)):
        kq, kQ, kZ, kY = agg_df.index.values[i]
        df = pd.read_csv(world_name+'/'+result_filename)
        df = df[df["kq"]==float(kq)]
        df = df[df["kQ"]==float(kQ)]
        df = df[df["kZ"]==float(kZ)]
        df = df[df["kY"]==float(kY)]

        avg_coverage = np.zeros(max_len)
        count = 0
        for draw_line in df["coverage_array"].values:
            count += 1
            dl = np.array(eval(draw_line))
            dl = np.pad(dl, ((0,max_len - len(dl))), mode='constant', constant_values=dl[-1])
            avg_coverage += dl
        if [kq, kQ, kZ, kY] == [-1, -1, -1, -1]: # strictly_constrained
            ax1.plot(avg_coverage/count,c="black")
        elif [kq, kQ, kZ, kY] == [0, 0, 0, 100]: # unconstrained
            ax1.plot(avg_coverage/count,c="orange")
        elif [kq, kQ, kZ, kY] == [0.8, 0, 0, 0]: # time pref
            ax1.plot(avg_coverage/count,c="red")
        elif [kq, kQ, kZ, kY] == [500, 0, 0, -100]: # multi objective
            ax1.plot(avg_coverage/count,c="deepskyblue")
        elif [kq, kQ, kZ, kY] == [0.005,1,0,100]:  # queue stabilizing
            ax1.plot(avg_coverage/count,c="green")
        elif [kq,kQ,kZ,kY] == [0.99,0,0,0]:
            ax1.plot(avg_coverage/count,c="red",linestyle="dashed")
        elif [kq,kQ,kZ,kY] == [1000,0,0,100]:
            ax1.plot(avg_coverage/count,c="green",linestyle="dashed")

    custom_lines = [(Line2D([0], [0], color="green",linestyle="solid"),Line2D([0], [0], color="green",linestyle="dashed")),
                (Line2D([0], [0], color="red",linestyle="solid"),Line2D([0], [0], color="red",linestyle="dashed")),
                Line2D([0], [0], color="deepskyblue"),
                Line2D([0], [0], color="orange"),
                Line2D([0], [0], color="black")]
    fig1.legend(custom_lines,["QS","TP","MO","UN","SC"], ncol=1, bbox_to_anchor=(0.95,0.75), numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
    fig1.tight_layout()
    plt.savefig(world_name+"/"+result_filename[:-4]+"_comparison.jpg")
    # plt.show()
    return

def plot_trade(world_name,result_filename,world_size):
    df = pd.read_csv(world_name+'/'+result_filename)
    agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
    agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
    agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
    agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
    agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
    fig2, ax2 = plt.subplots(1,1,figsize=(8,4))

    best_coverage = None
    unconstrained = []
    strictly_constrained = []
    time_preference = []
    multiobjective = []
    queue_stabilizing = []

    for dp in range(len(agg_df)):
        if agg_df.index.values[dp] == (0,0,0,0) or agg_df.index.values[dp] == (0,0,0,-100):
            unconstrained.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
            # ax2.axvline(agg_df["coverage"].values[dp],linestyle="--",c="black")
        elif agg_df.index.values[dp] == (-1,-1,-1,-1):
            strictly_constrained.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
            # ax2.axhline(agg_df["percent_localized"].values[dp],linestyle="--",c="black")
        elif agg_df.index.values[dp][-1] == 0:
            time_preference.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
        elif agg_df.index.values[dp][-1] < 0:
            multiobjective.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
        elif agg_df.index.values[dp][0] > 0 and agg_df["coverage"].values[dp] < 818:
            queue_stabilizing.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])

    mo = ax2.scatter(np.array(multiobjective)[:,0],np.array(multiobjective)[:,1],c="deepskyblue",marker="d",label="MO")
    sc = ax2.scatter(np.array(strictly_constrained)[:,0],np.array(strictly_constrained)[:,1],c="black",label="SC")
    qs = ax2.scatter(np.array(queue_stabilizing)[:,0],np.array(queue_stabilizing)[:,1],c="green",marker="x",label="QS")
    un = ax2.scatter(np.array(unconstrained)[:,0],np.array(unconstrained)[:,1],c="orange",marker="s",label="UN")
    tp = ax2.scatter(np.array(time_preference)[:,0],np.array(time_preference)[:,1],c="red",marker="v",label="TP")
    ax2.set_xlabel("Map size at data sink (cells)")
    ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
    ax2.legend(ncol=2)
    fig2.tight_layout()
    plt.autoscale()
    plt.savefig(world_name+"/"+result_filename[:-4]+"_trade.jpg")
    # plt.show()
    return

def print_data(world_name,result_filename,world_size):
    df = pd.read_csv(world_name+'/'+result_filename)
    # df = df[df["kY"]==0]
    # df = df[df["kq"]==0.005]
    # df = df[df["kQ"]==1000]
    # df = df[df["kZ"]==0]
    # print(df)
    agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
    agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
    agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
    agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
    agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
    # agg_df = agg_df.sort_values("coverage")
    # agg_df = agg_df[agg_df["fiedler"]>1]
    # agg_df = agg_df[agg_df["coverage"]>646]
    print(agg_df)

    for metric in agg_df.columns:
        if metric == "time":
            pass
        elif metric == "CRB":
            print("min",metric)
            print(agg_df[agg_df[metric] == min(agg_df[metric])])
        elif metric == "max_queue" or metric == "fiedler":
            print("max",metric)
            print(agg_df[agg_df[metric] == max(agg_df[metric])])
            print("min",metric)
            print(agg_df[agg_df[metric] == min(agg_df[metric])])
        else:
            print("max",metric)
            print(agg_df[agg_df[metric] == max(agg_df[metric])])

    return agg_df


def plot_trade_ZorNot(world_name,result_filename,world_size):
        df = pd.read_csv(world_name+'/'+result_filename)
        agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
        agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
        agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
        agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
        agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
        fig2, ax2 = plt.subplots(1,1,figsize=(4,4))

        best_coverage = None
        constrain = []
        dont = []
        for dp in range(len(agg_df)):
            if agg_df.index.values[dp] == (0,0,0,1):
                pass
            elif agg_df.index.values[dp] == (-1,-1,-1,-1):
                pass
            elif agg_df.index.values[dp][-1] == 0:
                pass
            elif agg_df.index.values[dp][-1] < 0:
                pass
            else:
                if agg_df.index.values[dp][2] > 0:
                    constrain.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
                else:
                    dont.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])

        ax2.scatter(np.array(constrain)[:,0],np.array(constrain)[:,1],c="green",marker="x",label=r"$k_Z$")
        ax2.scatter(np.array(dont)[:,0],np.array(dont)[:,1],c="black",marker="v",label=r"$\neg k_Z$")
        ax2.set_xlabel("Map size at data sink (cells)")
        ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
        ax2.legend(loc="upper right")#prop={'size': 10})
        fig2.tight_layout()
        plt.autoscale()
        plt.savefig(world_name+"/"+result_filename[:-4]+"_trade_kZ.jpg")
        # plt.show()
        return

def plot_trade_QorNot(world_name,result_filename,world_size):
        df = pd.read_csv(world_name+'/'+result_filename)
        df = df[df["kY"]>0]
        # df = df[df["kZ"]==0]
        agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
        agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
        agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
        agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
        agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
        fig2, ax2 = plt.subplots(1,1,figsize=(4,4))

        best_coverage = None
        constrain = []
        constrainq = []
        both = []
        dont = []
        for dp in range(len(agg_df)):
            if agg_df.index.values[dp] == (0,0,0,1):
                pass
            elif agg_df.index.values[dp] == (-1,-1,-1,-1):
                pass
            elif agg_df.index.values[dp][-1] == 0:
                pass
            elif agg_df.index.values[dp][-1] < 0:
                pass
            else:
                if agg_df.index.values[dp][1] > 0 and agg_df.index.values[dp][0] == 0:
                    constrain.append([agg_df["coverage"].values[dp],agg_df["fiedler"].values[dp]])
                elif agg_df.index.values[dp][1] == 0 and agg_df.index.values[dp][0] > 0:
                    constrainq.append([agg_df["coverage"].values[dp],agg_df["fiedler"].values[dp]])
                elif agg_df.index.values[dp][1] > 0 and agg_df.index.values[dp][0] > 0:
                    both.append([agg_df["coverage"].values[dp],agg_df["fiedler"].values[dp]])
                else:
                    dont.append([agg_df["coverage"].values[dp],agg_df["fiedler"].values[dp]])

        ax2.scatter(np.array(constrainq)[:,0],np.array(constrainq)[:,1],c="orange",marker="x",label=r"$\neg k_Q, k_q$")
        ax2.scatter(np.array(both)[:,0],np.array(both)[:,1],c="yellowgreen",marker="x",label=r"$k_Q, k_q$")
        ax2.scatter(np.array(constrain)[:,0],np.array(constrain)[:,1],c="green",marker="v",label=r"$k_Q, \neg k_q$")
        ax2.scatter(np.array(dont)[:,0],np.array(dont)[:,1],c="black",marker="v",label=r"$\neg k_Q, \neg k_q$")
        ax2.set_xlabel("Map size at data sink (cells)")
        ax2.set_ylabel(r"Average Connectivity ${\lambda_2}_D$")
        ax2.legend()
        fig2.tight_layout()
        plt.autoscale()
        plt.savefig(world_name+"/"+result_filename[:-4]+"_trade_kQ.jpg")
        # plt.show()
        return

def connectivity_v_gain(world_name,result_filename, world_size):
        df = pd.read_csv(world_name+'/'+result_filename)
        print(df.corr()["average_fiedler"])
        print(df.corr()["average_CRB"])
        return


if __name__ == "__main__":
    #try:
    f = sys.argv[1]
    world_name = sys.argv[2]
    result_filename = sys.argv[3]
    if world_name == "more_area_world":
        world_size = 50*50
    else:
        world_size = 30*30

    if f == "compare_avg_coverage":
        compare_avg_coverage(world_name,result_filename,world_size)
    elif f == "plot_trade":
        plot_trade(world_name,result_filename,world_size)
    elif f == "plot_trade_QorNot":
        plot_trade_QorNot(world_name,result_filename,world_size)
    elif f == "plot_trade_ZorNot":
        plot_trade_ZorNot(world_name,result_filename,world_size)
    elif f == "connectivity_v_gain":
        connectivity_v_gain(world_name,result_filename,world_size)
    elif f == "print_data":
        print_data(world_name,result_filename,world_size)
    else:
        print("Function name not recognized.")
    except:
       print("Example usage: process_results.py {plot_time_preference, compare_avg_coverage, plot_trade, plot_trade_QorNot, print_data}, {world_name}, {result_filename}")
