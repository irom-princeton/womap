
#%%import argparse
import pickle
import numpy as np
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# TODO results are saved twice

test_name = "test2"
test_result_directory = f"/n/fs/robot-data/womap/experiment_results/{test_name}"

# load experiment info from json
with open(f"{test_result_directory}/experiment_info.json", "r") as f:
    experiment_info = json.load(f)
    
# load experiment results
with open(f"{test_result_directory}/results.pkl", "rb") as f:
    results = pickle.load(f)

#%%
# plot performance of different models on each of the tasks
task_id = 0
task_name = experiment_info[str(task_id)]
print("Task=", task_name)

distance_thresh = 0.2

# TODO
num_models = 4

model_metrics = []
model_names = ["5-500", "50-500", "5-200", "10-200"]
for model_id in range(num_models):
    # load result for that task
    task_result = results[(model_id, task_id)]
    num_tests = len(task_result)
    print("Num tests=", num_tests)
    
    
    score_success_reward_pred = []
    score_success_reward_true = []
    score_success_dist_to_target = []
    percent_closer_to_target = []
    
    
    fig, (ax_reward, ax_distance, ax_ttarget) = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
    for i, tres in enumerate(task_result):
        # TODO temp fix
        time_history = tres['timestamp_history'][::2]
        # Convert time stamps to relative time (time elapsed since start)
        time = [t - time_history[0] for t in time_history]
        reward_t = tres['true_reward_history']
        reward_p = tres['reward_predicted_history']
        total_dist = tres['total_distance_travelled_history']
        target_dist = tres['distance_to_target_history'][1:]
        
        def find_target_time(arr, k):
            for i, val in enumerate(arr):
                if val > k:
                    return time_history[i]
            return None
          
        def find_target_time_reverse(arr, k):
            for i, val in enumerate(arr):
                if val < k:
                    return time_history[i]
            return None
        
        speed = 0.05
        best_time = target_dist[0] / speed
        
        def get_score(t):
          if t is None:
            return 0
          return best_time / t
        
        score_success_reward_pred.append(get_score(find_target_time(reward_p, 0.6)))
        score_success_reward_true.append(get_score(find_target_time(reward_t, 0.6)))
        score_success_dist_to_target.append(get_score(find_target_time_reverse(total_dist, 0.3)))
        percent_closer_to_target.append(1-target_dist[-1]/target_dist[0])
        
        
        ax_reward.plot(time, reward_t, label=f'Experiment {i}')
        ax_distance.plot(time, total_dist, label=f'Experiment {i}')
        ax_ttarget.plot(time, target_dist, label=f'Experiment {i}')

    
    # calculate average across tasks
    avg_score_success_reward_pred = np.mean(score_success_reward_pred)
    avg_score_success_reward_true = np.mean(score_success_reward_true)
    avg_score_success_dist_to_target = np.mean(score_success_dist_to_target)
    avg_percent_closer_to_target = np.mean(percent_closer_to_target)
    model_metrics.append(
      [avg_score_success_reward_pred,
       avg_score_success_reward_true,
       avg_score_success_dist_to_target,
       avg_percent_closer_to_target]
    )
    
    ax_reward.set_xlabel("Time (s)")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend()
    ax_reward.set_title("True/predicted reward")

    ax_distance.set_xlabel("Time (s)")
    ax_distance.set_ylabel("m")
    ax_distance.legend()
    ax_distance.set_title("Distance Travelled")
    
    ax_ttarget.set_xlabel("Time (s)")
    ax_ttarget.set_ylabel("m")
    ax_ttarget.legend()
    ax_ttarget.set_title("Dist to Target")

    plt.tight_layout()
    # plt.savefig(f"/n/fs/ap-project/active_perception_world_models/result_analysis/{task_name}_model_{model_id}.png")

# bar plot of 4 metrics
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()
# bar plot of 4 metrics
metrics = ["Reward Predicted", "Reward True", "Distance to Target", "Percent Closer to Target"]
for i, metric in enumerate(metrics):
    ax[i].bar(model_names, [model_metric[i] for model_metric in model_metrics])
    ax[i].set_title(metric)
    ax[i].set_ylabel("Score")
    ax[i].set_xlabel("Model")
    ax[i].set_ylim(0, 1.0)
    ax[i].grid()
    
# set the title of the figure
plt.suptitle(f"Task {task_name} - Model Comparison")
plt.tight_layout()
plt.savefig(f"/n/fs/ap-project/active_perception_world_models/result_analysis/{task_name}_model_comparison.png")
# for i, model_metric in enumerate(model_metrics):
print(task_name)
# %%
