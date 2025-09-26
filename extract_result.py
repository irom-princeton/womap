
#%% 
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from src.utils.result_utils import *


#%% 
# TODO define what data to extract
# ===== Final ===== #
# environments = ["kitchen_7_1_4_12"]
environments = ["kitchen_12_0410"]
planners = ["vlm-only-n_prop=1_query_mode=atomic-base",
            "wm-grid-reint_thresh=0.0_n_prop=1_default_action_init=random",
            "wm-vlm-reint_thresh=0.25_n_prop=3_query_mode=atomic-base_default_action_init=random"]
world_models = ["50-500"]
tasks = ["n=10_unseen_0424_easy", "n=10_unseen_0424_medium", "n=10_unseen_0424_hard"]

#%% 
# # TODO define what data to extract
# # ===== Kitchen 7 ===== #
# environments = ["kitchen_7_1_4_12"]
# planners = ["wm-grid-reint_thresh=0.0_n_prop=1_default_action_init=zeros"]
# world_models = ["50-500", "20-500", "10-500", "5-500", "1-300", "2-300", "5-300"]
# tasks = ["n=50_unseen_v1_easy", "n=50_unseen_v1_medium", "n=50_unseen_v1_hard"]

#%% 
# # ===== Kitchen 12 ===== #
# environments = ["kitchen_12_0410"]
# planners = ["wm-grid-reint_thresh=0.0_n_prop=1"]
# world_models = ["50-500"]
# tasks = ["n=50_unseen_v1_easy", "n=50_unseen_v1_medium", "n=50_unseen_v1_hard"]

#%%
# define parameters
success_thresh_reward_gt = 0.7
success_thresh_reward_pr = 0.6
success_thresh_dist = 0.2
speed = 0.05

#%%

# Find result paths based on provided keys
def find_result_paths(environment, planner, world_model, task):
    if "wm" not in planner:
        return f"/n/fs/robot-data/womap/everything_results/{environment}/{planner}/{task}/result.pkl"
    return f"/n/fs/robot-data/womap/everything_results/{environment}/{planner}/{world_model}/{task}/result.pkl"

for environment in environments:
    for planner in planners:
        for world_model in world_models:
            for task in tasks:
                result_path = find_result_paths(environment, planner, world_model, task)
                results = None
                with open(result_path, 'rb') as f:
                    results = pickle.load(f)
                
                result_metrics = {
                    "success_rate_reward_gt": [],
                    "success_rate_reward_pr": [],
                    "success_rate_distance": [],
                    "success_score_reward_gt": [],
                    "success_score_reward_pr": [],
                    "success_score_distance": [],
                    "confidence_increase": [],
                    "distance_percent_closer": []
                }
                
                # TODO success rate vs. 
                for i in range(len(results)):
                    result = results[i]
                    
                    best_time = (result['distance_to_target_history'][0] - 0.2) / speed
                    
                    success_time_reward_gt = get_success_time_inc(result['true_reward_history'], success_thresh_reward_gt, result['timestamp_history'])
                    success_time_reward_pr = get_success_time_inc(result['reward_predicted_history'], success_thresh_reward_pr, result['timestamp_history'])
                    success_time_distance = get_success_time_dec(result['distance_to_target_history'][0:-1], success_thresh_dist, result['timestamp_history'])
                    
                    # success_distance = get_success_distance(result['distance_to_target_history'][1:], success_thresh_dist, result['timestamp_history'])
                    
                    success_reward_gt = 1 if success_time_reward_gt is not None else 0
                    success_reward_pr = 1 if success_time_reward_pr is not None else 0
                    success_distance = 1 if success_time_distance is not None else 0
                    
                    # append to metrics
                    result_metrics["success_rate_reward_gt"].append(success_reward_gt)
                    result_metrics["success_rate_reward_pr"].append(success_reward_pr)
                    result_metrics["success_rate_distance"].append(success_distance)
                    
                    # print("best_time=", best_time)
                    # print("success_time_reward_gt=", success_time_reward_gt)
                    # print("success_time_reward_pr=", success_time_reward_pr)
                    # print("success_time_distance=", success_time_distance)
                    
                    
                    # calculate score
                    success_score_reward_gt = linear_score_scaling_by_time(success_time_reward_gt, best_time)
                    success_score_reward_pr = linear_score_scaling_by_time(success_time_reward_pr, best_time)
                    success_score_distance = linear_score_scaling_by_time(success_time_distance, best_time)
                    
                    success_score_distance = linear_score_scaling_by_time(success_time_distance, best_time)
                    
                    # append to metrics
                    result_metrics["success_score_reward_gt"].append(success_score_reward_gt)
                    result_metrics["success_score_reward_pr"].append(success_score_reward_pr)
                    result_metrics["success_score_distance"].append(success_score_distance)
                    
                    # print(success_distance, success_time_distance, best_time, success_score_distance)
                    # calculate confidence percent increase
                    start_gt_confidence = result['true_reward_history'][0]
                    end_gt_confidence = result['true_reward_history'][-1]
                    
                    start_target_distance = result['distance_to_target_history'][1]
                    end_target_distance = result['distance_to_target_history'][-1]
                    
                    result_metrics["confidence_increase"].append(max([(end_gt_confidence - start_gt_confidence) , 0]))
                    result_metrics["distance_percent_closer"].append(max([(start_target_distance - end_target_distance) / start_target_distance, 0]))
                    
                    
                print(f"Results for {environment}, {planner}, {world_model}, {task}:")
                print(result_metrics)
                
                # calculate average values
                avg_success_rate_reward_gt = np.mean(result_metrics["success_rate_reward_gt"])
                avg_success_rate_reward_pr = np.mean(result_metrics["success_rate_reward_pr"])
                avg_success_rate_distance = np.mean(result_metrics["success_rate_distance"])
                
                cleaned_success_score_reward_gt = [float(x) for x in result_metrics["success_score_reward_gt"]]
                avg_success_score_reward_gt = np.mean(cleaned_success_score_reward_gt)
                cleaned_success_score_reward_pr = [float(x) for x in result_metrics["success_score_reward_pr"]]
                avg_success_score_reward_pr = np.mean(cleaned_success_score_reward_pr)
                cleaned_success_score_distance = [float(x) for x in result_metrics["success_score_distance"]]
                avg_success_score_distance = np.mean(cleaned_success_score_distance)
                avg_confidence_increase = np.mean(result_metrics["confidence_increase"])
                cleaned_distance_percent_closer = [float(x) for x in result_metrics["distance_percent_closer"]]
                avg_distance_percent_closer = np.mean(cleaned_distance_percent_closer)
                
                print(f"Average success rate reward gt: {avg_success_rate_reward_gt}")
                # print(f"Average success rate reward pr: {avg_success_rate_reward_pr}")
                print(f"Average success rate distance: {avg_success_rate_distance}")
                # print(f"Average success score reward gt: {avg_success_score_reward_gt}")
                # print(f"Average success score reward pr: {avg_success_score_reward_pr}")
                print(f"Average success score distance: {avg_success_score_distance}")
                # print(f"Average confidence increase: {avg_confidence_increase}")
                # print(f"Average distance percent closer: {avg_distance_percent_closer}")
                print("\n")
        # breakpoint()        
# ['test_id', 'timestamp_history', 'true_reward_history', 'reward_predicted_history', 'total_distance_travelled_history', 'distance_to_target_history', 'angle_to_target_history', 'video_name']
# %%
