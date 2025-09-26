import json
import argparse
import subprocess
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        help="path to your experiment config file"
    )
    
    argparser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output toplevel directory"
    )
    
    argparser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="number of seconds to run the experiment"
    )
    
    args = argparser.parse_args()
    exp_config_path = args.exp_config
    output_dir = args.output_dir
    time_limit = args.time_limit
    
    # load experiment json config
    with open(exp_config_path, "r") as f:
        exp_config = json.load(f)
        
    environment_type = exp_config["environment_type"]
    environment = exp_config["environment"]
    
    # ===== LOADING PATHS ===== #
    
    experiment_dir = f"/n/fs/robot-data/womap/experiments/{environment}"
    
    # ===== LOADING MODELS AND EXPERIMENTS ===== #
    
    all_model_dicts = exp_config["models"]
    
    all_experiments = exp_config["experiments"] # if 'all_experiments' is empty, run all experiments
    
    if len(all_experiments) == 0:
        # Default: run all experiments
        all_experiments = [f for f in os.listdir(experiment_dir) if f.endswith(".json")]
        # remove the ".json" extension from the file names
        all_experiments = [f[:-5] for f in all_experiments]
        
    
    for model_id in range(len(all_model_dicts)):
        for experiment in all_experiments:
            
            experiment_file_path = f"{experiment_dir}/{experiment}.json"

            # Create a job name
            job_name = "exp0"

            # Optional: pass the parameters to your job via environment vars or arguments
            cmd = [
                "sbatch",
                "--job-name", job_name,
                "--export", f"exp_config={exp_config_path},output_dir={output_dir},time_limit={time_limit},model_id={model_id},exp_file_path={experiment_file_path}",
                "bash_scripts/run_single_experiment.bash"
            ]
            print("Submitting:", " ".join(cmd))
            subprocess.run(cmd)