# To retrain the model:

### Set up
* Follow the instructions in  `ppo-dash-training\README.md` to set up a conda envrionment and install the correct Obsticle Tower Envrionment for you platform
 
### Train 1st 100m steps
* `cd ppo-dash-training`
* `cd pytorch-a2c-ppo-acktr-gail`
* `python run.py --num-env-steps=50000000  --env=ObtRetro-v7 --num-processes=32 --num-mini-batch=8  --exp_name=cur047`
### Train 2nd 100m steps
* `python run.py --num-env-steps=50000000  --env=ObtRetro-v7 --num-processes=32 --num-mini-batch=8  --exp_name=cur047 --load --ppo-epoch=1 --lr=1-e4`
 
 
# Reproduce running the trained model against the validation seeds
* Using the folder `ppo-dash-validation` - this contains the trained model used for the submission for the paper.
* Follow the [obstacle-tower-challenge](https://github.com/Unity-Technologies/obstacle-tower-challenge#run-docker-image) instructions regarding running the docker image. 
Note: I have tested this on MacBook Pro and it works. As it builds, runs from docker images it should work on any platform.
 
# Reproduce running the trained model against the test seeds
* You can contact the Unity team otc@unity3d.com to ask if this is possible / validate our claim that we are 2nd place.
 
# Reproduce ‘Study of how individual elements impact learning’
* From the folder `ppo-dash-training`
* Follow the instructions in readme to set up a conda envrionment and install the correct Obsticle Tower Envrionment for you platform
 
### 001_baseline
* `cd 001_baseline`
* `python run.py  --num-processes=50  --exp_name=000_baseline-01
python run.py  --num-processes=50  --exp_name=000_baseline-02
python run.py  --num-processes=50  --exp_name=000_baseline-03`
 
### 002_reduce_action_space
* `cd 002_reduce_action_space`
* `python run.py  --num-processes=50  --exp_name=002_reduce_action_space-02`
* `python run.py  --num-processes=50  --exp_name=002_reduce_action_space-03`
* `python run.py  --num-processes=50  --exp_name=002_reduce_action_space-04`
 
### 003_recurrent
* `cd 003_recurrent`
* `python run.py  --num-processes=50  --exp_name=003_recurrent-01`
* `python run.py  --num-processes=50  --exp_name=003_recurrent-02`
* `python run.py  --num-processes=50  --exp_name=003_recurrent-03`
 
### 005_large_scale_hyperparms
* `cd 005_large_scale_hyperparms`
* `python run.py  --num-mini-batch=4 --exp_name=005_large_scale_hyperparms-01`
* `python run.py  --num-mini-batch=4 --exp_name=005_large_scale_hyperparms-02`
* `python run.py  --num-mini-batch=4 --exp_name=005_large_scale_hyperparms-03`
 
### 006_reduced_frame_stack
* `cd 006_reduced_frame_stack`
* `python run.py  --num-processes=50  --exp_name=006_reduced_frame_stack-01`
* `python run.py  --num-processes=50  --exp_name=006_reduced_frame_stack-02`
* `python run.py  --num-processes=50  --exp_name=006_reduced_frame_stack-03`
 
### 007_reduced_action_space_and_frame_stack
* `cd 007_reduced_action_space_and_frame_stack`
* `python run.py  --num-processes=50  --exp_name=007_reduced_action_space_and_frame_stack-01`
* `python run.py  --num-processes=50  --exp_name=007_reduced_action_space_and_frame_stack-02`
* `python run.py  --num-processes=50  --exp_name=007_reduced_action_space_and_frame_stack-03`

### 008_ra+rf+lshp
* `cd 008_ra+rf+lshp`
* `python run.py  --exp_name=008_ra+rf+lshp-01`
* `python run.py  --exp_name=008_ra+rf+lshp-02`
* `python run.py  --exp_name=008_ra+rf+lshp-03`

### 009_ra+rf+lshp+recurrent
* `cd 009_ra+rf+lshp+recurrent`
* `python run.py  --exp_name=009_ra+rf+lshp+recurrent-01`
* `python run.py  --exp_name=009_ra+rf+lshp+recurrent-02`
* `python run.py  --exp_name=009_ra+rf+lshp+recurrent-03`

### 010_ra+rf+lshp+recurrent+vec_obs
* `cd 010_ra+rf+lshp+recurrent+vec_obs`
* `python run.py  --exp_name=010_ra+rf+lshp+recurrent+vec_obs-01`
* `python run.py  --exp_name=010_ra+rf+lshp+recurrent+vec_obs-02`
* `python run.py  --exp_name=010_ra+rf+lshp+recurrent+vec_obs-03`

### 011_ra+rf+lshp+recurrent+vec_obs+norm_obs
* `cd 011_ra+rf+lshp+recurrent+vec_obs+norm_obs`
* `python run.py  --exp_name=011_ra+rf+lshp+recurrent+vec_obs+norm_obs-01`
* `python run.py  --exp_name=011_ra+rf+lshp+recurrent+vec_obs+norm_obs-02`
* `python run.py  --exp_name=011_ra+rf+lshp+recurrent+vec_obs+norm_obs-03`
 
### 012_ra+no_stack+lshp+recurrent+vec_obs+norm_obs
* `cd 012_ra+no_stack+lshp+recurrent+vec_obs+norm_obs`
* `python run.py  --exp_name=012_ra+no_stack+lshp+recurrent+vec_obs+norm_obs-01`
* `python run.py  --exp_name=012_ra+no_stack+lshp+recurrent+vec_obs+norm_obs-02`
* `python run.py  --exp_name=012_ra+no_stack+lshp+recurrent+vec_obs+norm_obs-03`

### 013_ra+no_stack+lshp+recurrent+vec_obs+norm_obs+rew_hacking
* `cd 013_ra+no_stack+lshp+recurrent+vec_obs+norm_obs+rew_hacking`
* `--exp_name=013_ra+no_stack+lshp+recurrent+vec_obs+norm_obs+rew_hacking-01`
* `--exp_name=013_ra+no_stack+lshp+recurrent+vec_obs+norm_obs+rew_hacking-02`
* `--exp_name=013_ra+no_stack+lshp+recurrent+vec_obs+norm_obs+rew_hacking-03`
 
