import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib.lines import Line2D

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
            # ax1.plot(eval(draw_line),c="b")
            dl = np.array(eval(draw_line))
            dl = np.pad(dl, ((0,max_len - len(dl))), mode='constant', constant_values=dl[-1])
            avg_coverage += dl
        if [kq, kQ, kZ, kY] == [-1, -1, -1, -1]:
            # ax1.plot(avg_coverage/count,label="strictly constrained")
            ax1.plot(avg_coverage/count,c="black")
        # elif [kq, kQ, kZ, kY] == [0, 0, 0, 100]:
            # ax1.plot(avg_coverage/count,label="unconstrained")
            # ax1.plot(avg_coverage/count,c="orange")
        # elif [kq, kQ, kZ, kY] == [0.35, 0, 0, 0] or [kq, kQ, kZ, kY] == [0.99,0,0,0]:
        elif [kq, kQ, kZ, kY] == [0.35, 0, 0, 0]:
            # ax1.plot(avg_coverage/count,label="time pref: "+str(kq))
            ax1.plot(avg_coverage/count,c="red")
        # elif [kq, kQ, kZ, kY] == [10, 0.1, 0.1, -100] or [kq, kQ, kZ, kY] == [1000, 0.1, 0.1, -100]:
        elif [kq, kQ, kZ, kY] == [10, 0.1, 0.1, -100]:
            # ax1.plot(avg_coverage/count,label="MO: "+str([kq, kQ, kZ, kY]))
            ax1.plot(avg_coverage/count,c="blue")
        # elif [kq, kQ, kZ, kY] == [0.005,0,0,100] or [kq, kQ, kZ, kY] == [0.001, 1000, 10, 100]:
        elif [kq, kQ, kZ, kY] == [0.005,0,0,100]:
            # ax1.plot(avg_coverage/count,label="queue: "+str([kq, kQ, kZ, kY]))
            ax1.plot(avg_coverage/count,c="green")

    custom_lines = [Line2D([0], [0], color="green"),
                # Line2D([0], [0], color="orange"),
                Line2D([0], [0], color="red"),
                Line2D([0], [0], color="blue"),
                Line2D([0], [0], color="black")]
    fig1.legend(custom_lines,["queue-stabilizing","time preference","multi-objective","strictly constrained"], bbox_to_anchor=(1,0.75))
    # fig1.legend(loc="lower right")
    # fig1.legend(bbox_to_anchor=(1.5,0.5))
    # fig1.legend(loc=7)
    # fig1.tight_layout()
    # fig1.subplots_adjust(right=0.75)
    fig1.tight_layout()
    plt.autoscale()
    plt.savefig(world_name+"/"+result_filename[:-4]+"_comparison_edited.jpg")
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
        if agg_df.index.values[dp] == (0,0,0,0):
            unconstrained.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
            # ax2.axvline(agg_df["coverage"].values[dp],linestyle="--",c="black")
        elif agg_df.index.values[dp] == (-1,-1,-1,-1):
            strictly_constrained.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
            # ax2.axhline(agg_df["percent_localized"].values[dp],linestyle="--",c="black")
        elif agg_df.index.values[dp][-1] == 0:
            time_preference.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
        elif agg_df.index.values[dp][-1] < 0:
            multiobjective.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])
        else:
            queue_stabilizing.append([agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]])

    mo = ax2.scatter(np.array(multiobjective)[:,0],np.array(multiobjective)[:,1],c="blue",marker="d",label="multi-objective")
    qs = ax2.scatter(np.array(queue_stabilizing)[:,0],np.array(queue_stabilizing)[:,1],c="green",marker="x",label="queue stabilizing")
    un = ax2.scatter(np.array(unconstrained)[:,0],np.array(unconstrained)[:,1],c="orange",marker="s",label="unconstrained")
    sc = ax2.scatter(np.array(strictly_constrained)[:,0],np.array(strictly_constrained)[:,1],c="black",label="strictly constrained")
    tp = ax2.scatter(np.array(time_preference)[:,0],np.array(time_preference)[:,1],c="red",marker="v",label="time preference")
    ax2.set_xlabel("Map size at data sink (cells)")
    ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
    # ax2.set_xlim((50,170))
    # cb = fig2.colorbar(sc)
    # cb.set_label(r"$\lambda_{2}$")
    # ax2.legend(ncol=2)#prop={'size': 10})
    l1 = ax2.legend([mo,sc],["multi-objective","strictly constrained"],loc="lower left")
    l2 = ax2.legend([qs, un, tp],["queue-stabilizing","unconstrained","time preference"],loc="upper right")
    plt.gca().add_artist(l1)
    # ax2.set_title("Time localized vs Cells visited")
    fig2.tight_layout()
    plt.autoscale()
    plt.savefig(world_name+"/"+result_filename[:-4]+"_trade.jpg")
    # plt.show()
    return

def print_data(world_name,result_filename,world_size):
    df = pd.read_csv(world_name+'/'+result_filename)
    # df = df.sort_values("kQ",ascending=False)
    # df = df[df["kY"]==-100]
    # print(df)
    agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
    agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
    agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
    agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
    agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
    agg_df = agg_df.sort_values("percent_localized")
    # agg_df = agg_df[agg_df["coverage"]>300]
    # agg_df = agg_df[agg_df["coverage"]<350]
    print(agg_df)

    for metric in agg_df.columns:
        if metric == "time":
            pass
        elif metric == "CRB":
            print("min",metric)
            print(agg_df[agg_df[metric] == min(agg_df[metric])])
        elif metric == "max_queue":
            print("max",metric)
            print(agg_df[agg_df[metric] == max(agg_df[metric])])
            print("min",metric)
            print(agg_df[agg_df[metric] == min(agg_df[metric])])
        else:
            print("max",metric)
            print(agg_df[agg_df[metric] == max(agg_df[metric])])

    # agg_df = agg_df[agg_df["percent_localized"]>30]
    # agg_df = agg_df[agg_df["coverage"]>105]
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

        ax2.scatter(np.array(constrain)[:,0],np.array(constrain)[:,1],c="purple",label=r"$k_Z > 0$")
        ax2.scatter(np.array(dont)[:,0],np.array(dont)[:,1],c="teal",label=r"$k_Z =0$")
        ax2.set_xlabel("Map size at data sink (cells)")
        ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
        # ax2.set_xlim((50,170))
        # cb = fig2.colorbar(sc)
        # cb.set_label(r"$\lambda_{2}$")
        ax2.legend(loc="upper right")#prop={'size': 10})
        # ax2.set_title(r"Virtual Queue $Z(t)$")
        fig2.tight_layout()
        plt.autoscale()
        plt.savefig(world_name+"/"+result_filename[:-4]+"_trade_kZ.jpg")
        # plt.show()
        return

def plot_trade_QorNot(world_name,result_filename,world_size):
        df = pd.read_csv(world_name+'/'+result_filename)
        # df = df[df["kZ"]==0]
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
                if agg_df.index.values[dp][1] > 0:
                    constrain.append([agg_df["coverage"].values[dp],agg_df["fiedler"].values[dp]])
                else:
                    dont.append([agg_df["coverage"].values[dp],agg_df["fiedler"].values[dp]])

        ax2.scatter(np.array(constrain)[:,0],np.array(constrain)[:,1],c="purple",label=r"$k_Q > 0$")
        ax2.scatter(np.array(dont)[:,0],np.array(dont)[:,1],c="teal",label=r"$k_Q = 0$")
        ax2.set_xlabel("Map size at data sink (cells)")
        ax2.set_ylabel(r"Average Connectivity $\lambda_2$")
        # ax2.set_xlim((50,170))
        # cb = fig2.colorbar(sc)
        # cb.set_label(r"$\lambda_{2}$")
        ax2.legend(loc="upper right")#prop={'size': 10})
        # ax2.set_title(r"Virtual Queue $Q(t)$")
        fig2.tight_layout()
        plt.autoscale()
        plt.savefig(world_name+"/"+result_filename[:-4]+"_trade_kQ.jpg")
        # plt.show()
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
    elif f == "print_data":
        print_data(world_name,result_filename,world_size)
    else:
        print("Function name not recognized.")
    #except:
    #    print("Example usage: process_results.py {plot_time_preference, compare_avg_coverage, plot_trade, plot_trade_QorNot, print_data}, {world_name}, {result_filename}, {fig_filename}")
