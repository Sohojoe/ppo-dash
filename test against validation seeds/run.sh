#!/bin/bash

# Use This line would work in case of the docker container
# Else the relevant env variables will not be loaded during evaluatino
source activate base
echo "root"


# pip install -e ./obstacle-tower-env/.


# cd large-scale-curiosity/
# python run_evaluation.py
# cd ..

cd pytorch/
python run_evaluation.py
cd ..