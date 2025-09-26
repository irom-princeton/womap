#!/bin/bash

# experiment folder
export EXP_DIR="experiment_configs"

# experiment
export EXP_FOLDER="<name of folder name of file>"
export EXP_FILE="<name of the config file, e.g., exp.json>"
# output
export OUTPUT_DIR="<path to output directory>"


# experiment
export EXP_FOLDER="<name of folder name of file>"
# output
export OUTPUT_DIR="<path to output directory>"

EXP_PATH="${EXP_DIR}/${EXP_FOLDER}/${EXP_FILE}"

TIME_LIMIT=100

# submit experiments
python submit_experiments.py --exp_config "${EXP_PATH}" --output_dir "${OUTPUT_DIR}" --time_limit "${TIME_LIMIT}"