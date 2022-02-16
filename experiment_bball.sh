#!bin/bash
echo Running on $HOSTNAME
#source ~/opt/anaconda3/etc/profile.d/conda.sh

cfg_json="configs/bball_complete/bball_complete.json"

#conda activate rim
nohup python3 train_bball.py --cfg_json $cfg_json