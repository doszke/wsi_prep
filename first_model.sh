#!/bin/bash
#PBS -q main
#PBS -l walltime=50:00:00
#PBS -l select=1:ncpus=8:mem=64000mb
#PBS -l software=smallimg
#PBS -N main_control
#PBS -m be
echo ". /home/doszke/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate tensorflow_env
cd /home/doszke/panda
python3 ./model.py >& first_model.txt