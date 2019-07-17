# PPO-Dash
Code for reproducing the results found in [PPO Dash: Improving Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1907.06704)

# About
PPO-Dash is a modified version of the PPO algrothem that utalises the following optimizations and best practices:
* Action Space Reduction
* Frame Stack Reduction
* Large Scale Hyperparameters
* Vector Observations
* Normalized Observations
* Reward Hacking
* Recurrent Memory

PPO-Dash was able to solve the first 10 levels of the Obsticle Tower Enviroment without the need for demonstrations or curosity based algorthemic enhancements.

The version of PPO-Dash in the technical paper, [placed 2nd](https://blogs.unity3d.com/2019/05/15/obstacle-tower-challenge-round-2-begins-today/) in Round One of the [Obsticle Tower Challenge](https://www.aicrowd.com/challenges/unity-obstacle-tower-challenge) with an average score of 10. We were able to reproduce this score in Round Two of the challenge, with a minor modifiaction (randomizing the themes during in training). We [placed 4th overall](https://www.aicrowd.com/challenges/unity-obstacle-tower-challenge/leaderboards), with a score of 10.8 with the addition of demonstrations.

# Reproducing Results
To reproduce the results listed in the paper and for round one of the competition, see [ReproduceRound1](ReproduceRound1.md)

# Acknowlegements
This codebase derives from [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) - [#8258f95](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/commit/8258f95d6c1959d02c6a412415138b95c32837a0)

# Citation
If you use PPO-Dash in your research, we ask that you cite the [technical report](https://arxiv.org/abs/1907.06704) as a reference.
