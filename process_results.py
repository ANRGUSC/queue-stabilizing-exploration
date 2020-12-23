import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compare_avg_coverage(world_name,result_filename,world_size):
    # fig1, (ax1) = plt.subplots(1,1,figsize=(5,5))
    fig1, (ax1) = plt.subplots(1,1,figsize=(7,4))
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
        elif [kq, kQ, kZ, kY] == [0, 0, 0, 1]:
            # ax1.plot(avg_coverage/count,label="unconstrained")
            ax1.plot(avg_coverage/count,c="grey")
        elif [kQ, kZ, kY] == [0, 0, 0]:
            # ax1.plot(avg_coverage/count,label="time pref: "+str(kq))
            ax1.plot(avg_coverage/count,c="red")
        elif kY == -100:
            # ax1.plot(avg_coverage/count,label="MO: "+str([kq, kQ, kZ, kY]))
            ax1.plot(avg_coverage/count,c="blue")
        else:
            # ax1.plot(avg_coverage/count,label="queue: "+str([kq, kQ, kZ, kY]))
            ax1.plot(avg_coverage/count,c="green")

    # fig1.legend(loc="lower right")
    # fig1.legend(bbox_to_anchor=(1.5,0.5))
    # fig1.legend(loc=7)
    # fig1.tight_layout()
    # fig1.subplots_adjust(right=0.75)
    fig1.tight_layout()
    plt.autoscale()
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
    fig2, ax2 = plt.subplots(1,1) #,figsize=(8,4))

    best_coverage = None
    for dp in range(len(agg_df)):
        if agg_df.index.values[dp] == (0,0,0,1):
            # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]],marker="X",label="unconstrained")
            ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="grey",label="unconstrained")
            # ax2.axvline(agg_df["coverage"].values[dp],linestyle="--",c="black")
        elif agg_df.index.values[dp] == (-1,-1,-1,-1):
            # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]],marker="D",label="strictly constrained")
            ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="black",label="strictly constrained")
            # ax2.axhline(agg_df["percent_localized"].values[dp],linestyle="--",c="black")
        elif agg_df.index.values[dp][-1] == 0:
            # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]],marker="O",label="time preference")
            ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="red",label="time preference")
        elif agg_df.index.values[dp][-1] < 0:
            # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]],marker="O",label="multiobjective")
            ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="blue",label="multiobjective")
        else:
            # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]],marker="O",label="queue-stabilizing")
            ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="green",label="queue-stabilizing")

    ax2.set_xlabel("Map size at data sink (cells)")
    ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
    # ax2.set_xlim((50,170))
    # cb = fig2.colorbar(sc)
    # cb.set_label(r"$\lambda_{2}$")
    # ax2.legend()#prop={'size': 10})
    ax2.set_title("Time localized vs Cells visited")
    fig2.tight_layout()
    plt.autoscale()
    plt.savefig(world_name+"/"+result_filename[:-4]+"_trade.jpg")
    # plt.show()
    return

def print_data(world_name,result_filename,world_size):
    df = pd.read_csv(world_name+'/'+result_filename)
    # df = df.sort_values("kQ",ascending=False)
    # df = df[df["kq"]==0]
    agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
    agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
    agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
    agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
    agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
    agg_df = agg_df.sort_values("coverage")
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


def plot_trade_QorNot(world_name,result_filename,world_size):
        df = pd.read_csv(world_name+'/'+result_filename)
        agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
        agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
        agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
        agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
        agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
        fig2, ax2 = plt.subplots(1,1) #,figsize=(8,4))

        best_coverage = None
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
                if agg_df.index.values[dp][-3] > 0:
                    ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="red",label=r"constrain $\lambda_2$")
                else:
                    ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c="blue",label=r"not constrain $\lambda_2$")

        ax2.set_xlabel("Map size at data sink (cells)")
        ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
        # ax2.set_xlim((50,170))
        # cb = fig2.colorbar(sc)
        # cb.set_label(r"$\lambda_{2}$")
        # ax2.legend()#prop={'size': 10})
        ax2.set_title("Time localized vs Cells visited: Constrain connectivity?")
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
    elif f == "print_data":
        print_data(world_name,result_filename,world_size)
    else:
        print("Function name not recognized.")
    #except:
    #    print("Example usage: process_results.py {plot_time_preference, compare_avg_coverage, plot_trade, plot_trade_QorNot, print_data}, {world_name}, {result_filename}, {fig_filename}")
