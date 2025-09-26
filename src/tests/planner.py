from abc import ABC, abstractmethod
import numpy as np
import torch


class Planner(ABC):
    """
    Planner Base Class
    """
    def __init__(
        self, 
        action_generator,
        world_model,
        termination_reward_threshold,
        reinitialize_action_reward_threshold,
        device
    ):
        self.world_model = world_model
        self.device = device
        self.action_generator = action_generator
        self.termination_reward_threshold = termination_reward_threshold
        self.reinitialize_action_reward_threshold = reinitialize_action_reward_threshold

    @abstractmethod
    def get_actions(self, obs, pose=None):
        pass

    def get_proposal(self, obs, pose, num_proposals):
        """
        Generate pose proposals based on the observation and pose.

        Args:
            obs: Observation from the environment.
            pose: Current pose of the agent.
            num_proposals: Number of proposals to generate.

        Returns:
            pose_proposals: List of next pose proposals.
        """
        return self.action_generator.generate_action_proposals(obs, pose, num_proposals)
    

    # TODO is it necessary to interpolate actions at random step sizes? If not, 
    # why do we enforce varying step sizes at training time? And we need to align action speed for a fair evaluation
    
    def interpolate_action(self, action, min_step_size, max_step_size):
        """
        action is a list of 6 elements: [x, y, z, roll, pitch, yaw].
        """
        action = np.array(action)
        translation_magnitude = np.linalg.norm(action[:3])  # Compute translation magnitude
        rotation_magnitude = np.linalg.norm(action[3:])  # Compute rotation magnitude

        interpolated_actions = []
        accumulated_translation = np.zeros(3)
        accumulated_rotation = np.zeros(3)

        if translation_magnitude > 0:
            # Interpolate both translation and rotation based on translation step size
            while np.linalg.norm(accumulated_translation) < translation_magnitude:
                step_size = np.random.uniform(min_step_size, max_step_size)
                step_vector = (action[:3] / translation_magnitude) * step_size
                rotation_vector = (action[3:] / translation_magnitude) * step_size  # Scale rotation with translation step

                if np.linalg.norm(accumulated_translation + step_vector) > translation_magnitude:
                    step_vector = action[:3] - accumulated_translation  # Ensure final step lands exactly
                
                accumulated_translation += step_vector
                accumulated_rotation += rotation_vector  # Accumulate rotation
                interpolated_actions.append(np.concatenate([step_vector, rotation_vector]).tolist())

        elif rotation_magnitude > 0:
            # No translation, only interpolate rotation
            num_steps = int(np.ceil(rotation_magnitude / min_step_size))  # Ensure sufficient steps
            step_vector = action[3:] / num_steps  # Evenly divide rotation

            for _ in range(num_steps):
                accumulated_rotation += step_vector
                interpolated_actions.append(np.concatenate([np.zeros(3), step_vector]).tolist())

        return interpolated_actions
    
    def interpolate_action_sequence(self, actions, min_step_size=0.03, max_step_size=0.06):
        interpolated_actions = []
        for action in actions:
            interpolated_action = self.interpolate_action(action, min_step_size, max_step_size)
            interpolated_actions.extend(interpolated_action)
        return interpolated_actions
        