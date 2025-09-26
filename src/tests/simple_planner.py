import torch
from src.tests.planner import Planner

class SimplePlanner(Planner):
    """
    Does not evalute action proposals
    """
    def __init__(
        self, 
        action_generator, 
        world_model=None,
        termination_reward_threshold=0.8,
        reinitialize_action_reward_threshold=0.2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__(
            action_generator,
            world_model,
            termination_reward_threshold,
            reinitialize_action_reward_threshold,
            device
        )

    def get_actions(self, current_obs_img, target_obj_embed=None, curr_pose=None):
        
        action_proposal = self.action_generator.generate_action_proposals(
            current_obs_img, curr_pose, 1
        )
        
        if (action_proposal is None) or (len(action_proposal) == 0):
            return None
        
        # get interpolated action sequence
        action_sequence = self.interpolate_action_sequence(action_proposal)
        
        placeholder_sequence = [0] * len(action_sequence)
        
        return action_sequence, placeholder_sequence, placeholder_sequence