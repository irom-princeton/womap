import numpy as np


def get_success_time_inc(arr, k, time_history):
    for i, val in enumerate(arr):
        if val > k:
            return time_history[i]
    return None
    
def get_success_time_dec(arr, k, time_history):
    for i, val in enumerate(arr):
        if val < k:
            return time_history[i]
    return None

def get_success_distance(distance_traveled, distance_to_target, distance_threshold):
    for i, distance in enumerate(distance_to_target):
        if distance < distance_threshold:
            return distance_traveled[i]
    return None

def linear_score_scaling_by_time(test_time, best_time):
    if test_time == None:
        return 0
    return best_time / test_time