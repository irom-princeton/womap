#!/bin/bash

# Path to the SLURM script
job_script=bash_scripts/train_single_wm_example.bash

for i in 5 10 15 20; do # Number of scenes
    config_name="<name of the config file, without yaml extension>"
    echo "Submitting: $config_name"
    sbatch "$job_script" "$config_name"

done
