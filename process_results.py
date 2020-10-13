import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def process_results(world_name,result_filename,world_size,gains=None,fig_filename=None):
    df = pd.read_csv(world_name+'/'+result_filename)
    df = df.sort_values("kQ",ascending=False)
    # df = df[df["kq"]==0]
    # df = df[df["kQ"]>0]
    # df = df[df["kZ"]==0]

    if gains or False:
        fig1, (ax1) = plt.subplots(1,1,figsize=(4,4))
        fig1, (ax1) = plt.subplots(1,1,figsize=(5,5))
        for gains in [[-1,-1,-1,-1],[0,0,0,1],[0.1,0.001,0,1]]:
            df = pd.read_csv(world_name+'/'+result_filename)
            df = df[df["kq"]==float(gains[0])]
            df = df[df["kQ"]==float(gains[1])]
            df = df[df["kZ"]==float(gains[2])]
            df = df[df["kY"]==float(gains[3])]

            agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
            agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
            agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time"]*10
            agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time"]*10
            agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
            print(agg_df)
            ax1.axhline(world_size,c='black',linestyle='--')
            # ax1.set_title("Map cells visited")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Map size at data sink (cells)")

            avg_coverage = np.zeros(362)
            count = 0
            for draw_line in df["coverage_array"].values:
                count += 1
                # ax1.plot(eval(draw_line),c="b")
                dl = np.array(eval(draw_line))
                dl = np.pad(dl, ((0,362 - len(dl))), mode='constant', constant_values=dl[-1])
                avg_coverage += dl
            if gains == [-1, -1, -1, -1]:
                ax1.plot(avg_coverage/count,label="strictly constrained")
            elif gains == [0, 0, 0, 1]:
                ax1.plot(avg_coverage/count,label="unconstrained")
            else:
                ax1.plot(avg_coverage/count,label="queue-stabilizing")
        fig1.legend()
        fig1.tight_layout()
        # plt.show()
        plt.savefig("compare_avg_coverage.jpg")
        return

    agg_df = df.groupby(['kq','kQ','kZ','kY']).agg({'coverage':['mean'],'time':['mean','sum'],'time_connected':['sum'],'average_fiedler':['mean'],'time_localized':['sum'],'average_CRB':['mean'],'max_queue':['max']})
    agg_df.columns = ["coverage","time","time_total","time_connected","fiedler","time_localized","CRB","max_queue"]
    agg_df["percent_connected"] = agg_df["time_connected"]/agg_df["time_total"]*100
    agg_df["percent_localized"] = agg_df["time_localized"]/agg_df["time_total"]*100
    agg_df = agg_df.drop(["time_total","time_connected","time_localized"],1)
    # agg_df = agg_df.sort_values("coverage")
    # agg_df = agg_df.sort_values("percent_localized")
    agg_df = agg_df.sort_values("fiedler")

    # for metric in agg_df.columns:
    #     if metric == "max_queue":
    #         print("min",metric)
    #         print(agg_df[agg_df[metric] == min(agg_df[metric])])
    #     else:
    #         print("max",metric)
    #         print(agg_df[agg_df[metric] == max(agg_df[metric])])

    if True:
        # fig2, ax2 = plt.subplots(1,1,figsize=(8,4))
        fig2, ax2 = plt.subplots(1,1,figsize=(6,4))
        # print(agg_df.index.values)
        q_ids = []
        kQ_ids = []
        notkQ_ids = []
        # scatter_ids = []
        best_coverage = None
        for dp in range(len(agg_df)):
            if agg_df.index.values[dp] == (0,0,0,1):
                # scatter_ids.append(dp)
                # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]]) #,marker="v",label="unconstrained")
                # ax2.text(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp],"unconstrained")
                # ax2.annotate('unconstrained', xy=(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]), xytext=(90, 8), arrowprops=dict(arrowstyle="->"))
                # ax2.annotate('unconstrained', xy=(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]), xytext=(120, 8), arrowprops=dict(arrowstyle="->"))
                ax2.axvline(agg_df["coverage"].values[dp],linestyle="--",c="black")
            elif agg_df.index.values[dp] == (-1,-1,-1,-1):
                # scatter_ids.append(dp)
                # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]]) #,marker="D",label="strictly constrained")
                # ax2.text(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp],"strictly constrained")
                # ax2.annotate('strictly constrained', xy=(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]), xytext=(55, 15), arrowprops=dict(arrowstyle="->"))
                # ax2.annotate('strictly \n constrained', xy=(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp]), xytext=(80, 20), arrowprops=dict(arrowstyle="->"))
                ax2.axhline(agg_df["percent_localized"].values[dp],linestyle="--",c="black")
            elif agg_df.index.values[dp] == (0,0,0,0):
                pass
                # scatter_ids.append(dp)
                # ax2.scatter([agg_df["coverage"].values[dp]],[agg_df["percent_localized"].values[dp]],c=[agg_df["fiedler"].values[dp]],marker="^",label="random")
                # ax2.text(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp],"random")
            elif agg_df.index.values[dp][1] == 0:
                notkQ_ids.append(dp)
            else:
                # q_ids.append(dp)
                kQ_ids.append(dp)
                # scatter_ids.append(dp)
                # if agg_df["coverage"].values[dp] == max(agg_df["coverage"]):
                #     ax2.text(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp],str(agg_df.index.values[dp]))
                # elif agg_df["percent_localized"].values[dp] == max(agg_df["percent_localized"]):
                #     ax2.text(agg_df["coverage"].values[dp],agg_df["percent_localized"].values[dp],str(agg_df.index.values[dp]))

        # ax2.plot(agg_df["coverage"][q_ids],agg_df["percent_localized"][q_ids],c="grey")
        # sc = ax2.scatter(agg_df["coverage"][scatter_ids],agg_df["percent_localized"][scatter_ids],c=agg_df["fiedler"][scatter_ids],label="queue-stabilizing")
        sc = ax2.scatter(agg_df["coverage"][kQ_ids],agg_df["percent_localized"][kQ_ids],c="red",label=r"Constraining $\lambda_2$")
        sc = ax2.scatter(agg_df["coverage"][notkQ_ids],agg_df["percent_localized"][notkQ_ids],c="blue",label=r"Not constraining $\lambda_2$")

        ax2.set_xlabel("Map size at data sink (cells)")
        ax2.set_ylabel(r"Time below $\theta_{CRB}$ (\%)")
        # ax2.set_xlim((50,170))
        # cb = fig2.colorbar(sc)
        # cb.set_label(r"$\lambda_{2}$")
        ax2.legend(prop={'size': 10})
        # ax2.set_title("Efficiency v Accuracy trade")
        fig2.tight_layout()
        # plt.autoscale()

        # annot = ax2.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
        #             bbox=dict(boxstyle="round", fc="w"),
        #             arrowprops=dict(arrowstyle="->"))
        # annot.set_visible(False)
        #
        # def update_annot(ind):
        #
        #     pos = sc.get_offsets()[ind["ind"][0]]
        #     annot.xy = pos
        #     text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
        #                            " ".join([str(agg_df.index.values[scatter_ids][n]) for n in ind["ind"]]))
        #     annot.set_text(text)
        #     # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        #     annot.get_bbox_patch().set_alpha(0.4)
        #
        #
        # def hover(event):
        #     vis = annot.get_visible()
        #     if event.inaxes == ax2:
        #         cont, ind = sc.contains(event)
        #         if cont:
        #             update_annot(ind)
        #             annot.set_visible(True)
        #             fig2.canvas.draw_idle()
        #         else:
        #             if vis:
        #                 annot.set_visible(False)
        #                 fig2.canvas.draw_idle()
        #
        # fig2.canvas.mpl_connect("motion_notify_event", hover)

        # plt.savefig(result_filename[:-4]+".jpg")
        plt.savefig("4kQ_or_not.jpg")
        plt.show()

    # agg_df = agg_df[agg_df["percent_localized"]>30]
    # agg_df = agg_df[agg_df["percent_localized"]<32]
    # agg_df = agg_df[agg_df["coverage"]>103]
    # agg_df = agg_df[agg_df["coverage"]<150]
    # agg_df = agg_df[agg_df["coverage"]>105]
    return agg_df

if __name__ == "__main__":
    world_name = sys.argv[1]
    result_filename = sys.argv[2]
    if world_name == "obstacle_world" or world_name == "wall_world":
        world_size = 225
    elif world_name == "giant_world":
        world_size = 900
    else:
        print("What is the world size?")

    try:
        kq, kQ, kZ, kY = sys.argv[3:7]
        fig_filename = sys.argv[7]
        df = process_results(world_name,result_filename,world_size,[kq,kQ,kZ,kY],fig_filename)
    except ValueError:
        df = process_results(world_name,result_filename,world_size)
        print("Best results...")
        print(df)
