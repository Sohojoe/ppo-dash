# ppo-dash
Improving Generalization in Deep Reinforcement Learning


# install

## cloan this repro
```
... git cloan code
cd ppo-dash
```


## Set up conda environment
refer to [pytorch.org](https://pytorch.org) for the correct pytorch install for your hardware. 

For example: windows with GPU / CUDA10 

```
conda create -n ppo-dash python=3.6 pytorch torchvision cudatoolkit=10.0 -c pytorch
```

then

```
conda activate ppo-dash
conda env update --file environment.yml
```

## Download the Obsticle Tower Challenge environment for you hardware

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v1.3/obstacletower_v1.3_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v1.3/obstacletower_v1.3_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v1.3/obstacletower_v1.3_windows.zip |

For checksums on these files, see [here](https://storage.googleapis.com/obstacle-tower-build/v1.3/ote-v1.3-checksums.txt).



### ppo is based on [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) - [#8258f95](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/commit/8258f95d6c1959d02c6a412415138b95c32837a0)




