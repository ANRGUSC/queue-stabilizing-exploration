# A Queue-Stabilizing Framework for Multi-Robot Exploration

Motivated by planetary exploration, we consider
the problem of deploying a network of mobile robots to
explore an unknown environment and share information with
a stationary data sink. The configuration of robots affects both
network connectivity and the accuracy of relative localization.
Robots explore autonomously and can store data locally in
their queues. When a communication path exists to the data
sink, robots transfer their data. Because robots may fail
in a non-deterministic manner, causing loss of the data in
their queues, enabling communication is important. However,
strict constraints on connectivity and relative positions limit
exploration. To take a more flexible approach to managing these
multiple objectives, we use Lyapunov-based stochastic optimiza-
tion to maximize new information while using virtual queues to
constrain time-average expectations of metrics of interest. These
include queueing delay, network connectivity, and localization
uncertainty. The result is a distributed online controller which
autonomously and strategically breaks and restores connectivity
as needed. We explicitly account for obstacle avoidance, limited
sensing ranges, and noisy communication/ranging links with
line-of-sight occlusions. We use queuing theory to analyze the
average delay experienced by data in our system and guarantee
connectivity will be recovered when feasible. We demonstrate
in simulation that our queue-stabilizing controller can reduce
localization uncertainty and achieve better coverage than two
state of the art approaches.

A video summarizing this work in under 3 minutes can be found in the folder `RAL_multimedia`.  
(An older video summarizing a previous version can be found here [https://youtu.be/xKM0vvIVHX8].)

USAGE:

`python lyapunov_explore.py {world folder name} {configuration json file name}` runs a script to start simulations for several approaches and various values of queue-stabilizing gains, time preference target ratios, and multi-objective weights. Generates a .csv file of results.

`python process_results.py {desired function} {world folder name} {results csv file name}` runs the desired function to plot the average coverage or the localizability vs coverage trade, amongst other plots, for a given .csv file of results. Generates .jpg images of plots.

`grid_classes.py` contains the source code for the simulation.

`grid_utils.py` contains several additional utility functions.

VIEWING SIMULATIONS:

To view the outcome of a simulation, navigate to the world folder (e.g. `baseline_world`). Then navigate to one of the subfolders associated with a configuration file. The folder `gifs_configX` will contains .gif videos of one trial of the simulation.

Naming convention: Each .gif file is named by four numbers.   
[0, 0, 0, 0] indicates unconstrained exploration.   
[-1, -1 ,-1, -1] indicates strictly constrained exploration.   
[X, 0, 0, 0] indicates time preference exploration with rho = X.   
[X, Y, Z, W] with W \< 0 indicates multi-objective exploration with weights w1, w2, w3, w4 = X, Y, Z, -W.   
[X, Y, Z, W] with W \> 0 indicates queue-stabilizing exploration (our novel contribution) with gains kq, kQ, kZ, kY = X, Y, Z, W.  
  
Please reach out to lilliamc@usc.edu with any questions or issues.
