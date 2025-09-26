import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def get_action(prev_cam2world, current_cam2world):
    """
    Args:
        prev_cam2world (array-like): The before (3x4 or 4x4) view matrix (x_A^W)
        current_cam2world (array-like): The after (3x4 or 4x4) view matrix (x_B^W)

    Returns:
        action (array-like): A (6,) numpy vector representing action applied in prev_cam2world's local frame, where:
            - First 3 elements represent translation (x, y, z).
            - Last 3 elements represent rotation (roll, pitch, ywa) in radians.
    """

    is_torch = isinstance(prev_cam2world, torch.Tensor)

    if is_torch:
        prev_cam2world = prev_cam2world.detach().cpu().numpy()
        current_cam2world = current_cam2world.detach().cpu().numpy()

    # pose at the preceding timestep
    prev_pose = np.eye(4)
    prev_pose[:3] = prev_cam2world[:3]

    # pose at the current timestep
    current_pose = np.eye(4)
    current_pose[:3] = current_cam2world[:3]

    # compute the action matrix
    action = np.linalg.inv(prev_pose) @ current_pose # action = T_B^A

    # extract rotation and translation
    rot_action = R.from_matrix(action[:3, :3]).as_euler('XYZ')
    translation_action = action[:3, -1]

    action = np.concatenate((translation_action, rot_action))

    return torch.tensor(action) if is_torch else action


def apply_action(prev_cam2world, action):
    """
    Args:
        prev_cam2world (array-like): The before (3x4 or 4x4) view matrix (x_A^W)
        action (array-like): A (6,) array where:
            - First 3 elements represent translation (x, y, z).
            - Last 3 elements represent rotation (roll, pitch, yaw) in radians.

    Returns:
        new_cam2world (array-like): The (3x4) view matrix after applying action on prev_cam2world in its local frame.
    """

    is_torch = isinstance(prev_cam2world, torch.Tensor)

    # Convert action to NumPy for SciPy compatibility
    if is_torch:
        prev_cam2world = prev_cam2world.detach().cpu().numpy()
        
    is_torch_action = isinstance(action, torch.Tensor)
    if is_torch_action:
        action = action.detach().cpu().numpy()
        
    action = action.flatten()
    
    # TODO need to change back
    # xyz_vector = action[3:]
    # rot_vector = action[:3]
    xyz_vector = action[:3]
    rot_vector = action[3:]

    rot_mat = R.from_euler('XYZ', rot_vector).as_matrix()

    action_mat = np.eye(4)
    action_mat[:3, :3] = rot_mat
    action_mat[:3, -1] = xyz_vector

    pose = np.eye(4)
    pose[:3] = prev_cam2world[:3]

    # TODO need to change back
    # action_mat = np.linalg.inv(action_mat)
        
    new_pose = (pose @ action_mat)[:3]


    return torch.tensor(new_pose) if is_torch else new_pose

# if __name__ == "__main__":
#     # Example usage
#     prev_cam2world = [[-0.34802526, -0.20234635,  0.9153876 ,  1.3384867 ],
#         [ 0.93748516, -0.07511761,  0.33982193,  0.15355237],
#         [-0.0        ,  0.97642887,  0.21583952,  0.75961226]]
    
#     current_cam2world = [[-0.34802526, -0.20234635,  0.9153876 ,  1.5384867 ],
#         [ 0.93748516, -0.07511761,  0.33982193,  0.15355237],
#         [-0.0        ,  0.97642887,  0.21583952,  0.75961226]]
    
#     a = get_action(prev_cam2world, current_cam2world)
#     computed_cam2world = apply_action(prev_cam2world, a)
#     print("Action:")
#     print(a)
#     print("Computed camera to world matrix:")
#     print(computed_cam2world)
#     print("True camera to world matrix:")
#     print(current_cam2world)