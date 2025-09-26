#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --ntasks-per-node=16                            ## Processors per node
#SBATCH --mem=64G                                       ## RAM per node
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --gres=gpu:1                                    ## Number of GPUs
#SBATCH --job-name=19-k125                                 ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=<email_address>

# load modules or conda environments here
source ~/.bashrc

# run
cd <path to codebase>

config_name=$1

# export model name
export model=dino

# export decoder name
export decoder_model=vqvae

# No ablation

# frozen
python main_train.py --config_name ${config_name} \
--wandbproj $(date +%F)-world_model_multiscene \
--wandbexp ${model}-frozen-decoder-${decoder_model}-$(date "+%H:%M:%S")-${config_name} \
--enable_decoder