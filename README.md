# HydraNet
In tennis tournaments, momentum, a critical yet elusive phenomenon, reflects the dynamic shifts in performance of athletes that can decisively influence match outcomes. Despite its significance, momentum in terms of effective modeling and multi-granularity analysis across points, games, sets, and matches in tennis tournaments remains underexplored. In this study, we define a novel Momentum Score (MS) metric to quantify a player's momentum level in multi-granularity tennis tournaments, and design HydraNet, a momentum-driven state-space duality-based framework, to model MS by integrating thirty-two heterogeneous dimensions of athletes performance in serve, return, psychology and fatigue. HydraNet integrates a Hydra module, which builds upon a state-space duality (SSD) framework, capturing explicit momentum with a sliding-window mechanism and implicit momentum through cross-game state propagation. It also introduces a novel Versus Learning method to better enhance the adversarial nature of momentum between the two athletes at a macro level, along with a Collaborative-Adversarial Attention Mechanism (CAAM) for capturing and integrating intra-player and inter-player dynamic momentum at a micro level. Additionally, we construct a million-level tennis cross-tournament dataset spanning from 2012-2023 Wimbledon and 2013-2023 US Open, and validate the multi-granularity modeling capability of HydraNet for the MS metric on this dataset. Extensive experimental evaluations demonstrate that the MS metric constructed by the HydraNet framework provides actionable insights into how momentum impacts outcomes at different granularities, establishing a new foundation for momentum modeling and sports analysis. To the best of our knowledge, this is the first work to explore and effectively model momentum across multiple granularities in professional tennis tournaments.

# Requirements
  * Python 3.8 or higher
  * chardet==4.0.0
  * numpy==1.24.3
  * pandas==1.4.1
  * scikit_learn==1.3.0
  * scipy==1.10.1
  * torch==2.4.1
  * tqdm==4.66.5

# Data
  * WID.csv
  * USD.csv

# Running  the Code
  * Execute ```python HydraNet-point.py``` to run the code of point granularity.
  * Execute ```python HydraNet-game.py``` to run the code of game granularity.
  * Execute ```python HydraNet-set.py``` to run the code of set granularity.
  * Execute ```python HydraNet-match.py``` to run the code of match granularity.

# Note
```
 The running result of the model is saved in the form of txt file, and the weight file of the model is saved in the form of pt.
```

