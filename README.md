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

The video accompanying this work can be found here [add link].
