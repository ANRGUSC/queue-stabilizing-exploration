# A Queue-Stabilizing Framework for Multi-Robot Exploration

Motivated by planetary exploration, we consider
the problem of deploying a network of mobile robots to
explore an unknown environment and share information with
a stationary data sink. The configuration of robots affects both
the accuracy of relative localization and network connectivity.
Robots explore autonomously and can store data locally in
their queues. When a communication path exists to the data
sink, robots transfer their data. Because robots may fail
in a non-deterministic manner, causing loss of the data in
their queues, enabling communication is important. However,
strict constraints on connectivity and relative positions limit
exploration. To take a more flexible approach to managing
these multiple objectives, we use Lyapunov-based stochastic
optimization to maximize new information while using real and
virtual queues to constrain time average expectations of metrics
of interest. These include the size of the data queue, the network
connectivity, and the localization performance. The result is a
distributed online controller which autonomously and strategically 
breaks and restores connectivity as needed. We explicitly
account for obstacle avoidance, limited sensing ranges, and
noisy communication/ranging links with line-of-sight occlusions.
We demonstrate the performance of our controller in simulation
with special attention to local optima and limit cycles, and
show (1) up to 13% improvement in coverage over purely
information-theoretic unconstrained exploration and (2) a 2.87x
improvement in average connectivity and 28% improvement in
localizability relative to strictly constrained exploration.

The video accompanying this work can be found here [https://youtu.be/xKM0vvIVHX8].

USAGE:

`python lyapunov_explore.py {world folder name} {configuration json file name}` runs a script to start simulations for several approaches and various values of queue-stabilizing gains, time preference target ratios, and multi-objective weights. Generates a .csv file of results.

`python process_results.py {desired function} {world folder name} {results csv file name}` runs the desired function to plot the average coverage or the localizability vs coverage trade, amongst other plots, for a given .csv file of results. Generates .jpg images of plots.

`grid_classes.py` contains the source code for the simulation.

`grid_utils.py` contains several additional utility functions.

VIEWING SIMULATIONS

To view the outcome of a simulation, navigate to the world folder (e.g. `baseline_world`). Then navigate to one of the subfolders associated with a configuration file. The folder `gifs_configX` will contains .gif videos of one trial of the simulation.

Naming convention: Each .gif file is named by four numbers.   
[0, 0, 0, 0] indicates unconstrained exploration.   
[-1, -1 ,-1, -1] indicates strictly constrained exploration.   
[X, 0, 0, 0] indicates time preference exploration with rho = X.   
[X, Y, Z, W] with W \< 0 indicates multi-objective exploration with weights w1, w2, w3, w4 = X, Y, Z, -W.   
[X, Y, Z, W] with W \> 0 indicates queue-stabilizing exploration (our novel contribution) with gains kq, kQ, kZ, kY = X, Y, Z, W.  
  
Please reach out to lilliamc@usc.edu with any questions or issues.
