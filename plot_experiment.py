import argparse
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="experiment id",
    )
    
    args = parser.parse_args()
    raw_id = args.id
    # Load the results
    results_path = f"{raw_id}/results.pkl"
    saves_path = f"{raw_id}/results.png"

    with open(results_path, "rb") as f:
        results = pickle.load(f)
    
    # generate 30 distinct colors
    cmap = matplotlib.colormaps['tab10']
    colors = [cmap(i % 10) for i in range(30)]
    line_styles = ['-', '--', ':']

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Experiment Metrics")

    # Store handles and labels for a single combined legend
    all_handles = []
    all_labels = []
    
    breakpoint()

    # Iterate through the results and plot
    for experiment_key, all_experiments in results.items():
        planner_name, wm_model_name, wm_ckpt, test_config, i = experiment_key
        legend_prefix = f"{planner_name}_{wm_model_name}_{wm_ckpt}"
        
        for experiment_id, metric in enumerate(all_experiments):
            legend_prefix_experiment = f"{legend_prefix}_{experiment_id}"
            color = colors[experiment_id % len(colors)]

            # Plot the metrics
            h1, = ax[0].plot(
                metric["true_reward_history"], 
                label=legend_prefix_experiment + " True",
                linestyle=line_styles[0],
                color=color
                )
            h2, = ax[0].plot(
                metric["reward_predicted_history"], 
                label=legend_prefix_experiment + " Pred",
                linestyle=line_styles[1],
                color=color
                )
            h3, = ax[0].plot(
                np.arange(1, len(metric["next_reward_predicted_history"])),
                metric["next_reward_predicted_history"][:-1],
                label=legend_prefix_experiment + " Pred_-1",
                linestyle=line_styles[2],
                color=color
                )
            h4, = ax[1].plot(
                metric["distance_to_target_hitstory"], 
                label=legend_prefix_experiment,
                color=color
                )
            h5, = ax[2].plot(
                metric["angle_to_target_history"], 
                label=legend_prefix_experiment, 
                color=color
                )

            all_handles.extend([h5])
            all_labels.extend([h5.get_label()])

    ax[0].set_title("Reward History")
    ax[1].set_title("Distance to Target")
    ax[2].set_title("Angle to Target")

    for axis in ax:
        axis.grid()

    # Add shared legend at the bottom
    fig.legend(all_handles, all_labels, loc='lower center', ncol=3, fontsize='small')

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for suptitle and legend
    plt.savefig(saves_path)
    print(f"*** Visualization saved to {saves_path} ***")
        