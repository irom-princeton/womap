#!/bin/bash
#SBATCH --nodes=1                                       ## Node count
#SBATCH --ntasks-per-node=16                            ## Processors per node
#SBATCH --mem=64G                                       ## RAM per node
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --gres=gpu:1                                    ## Number of GPUs
#SBATCH --job-name=test                                 ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=<email_address>


source ~/.bashrc

# TODO change directory to path of your own repo

# cd ...
cd <path to codebase>


python run_single_experiment.py \
    --exp_config "$exp_config" \
    --output_dir "$output_dir" \
    --time_limit "$time_limit" \
    --model_id "$model_id" \
    --exp_file_path "$exp_file_path"