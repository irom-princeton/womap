import torch
# from pytorch3d import transforms as py3d_transforms
from src.tests.planner import Planner
from src.tests.action_generator import CEMActionGenerator
from collections import deque


class WorldModelPlanner(Planner):
    """
    Does not evalute action proposals
    """
    def __init__(
        self, 
        action_generator, 
        world_model,
        enable_action_damping=True,
        default_action_initialization="random", # random
        termination_reward_threshold=0.8,
        reinitialize_action_reward_threshold=0.2,
        num_proposals=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__(
            action_generator,
            world_model,
            termination_reward_threshold,
            reinitialize_action_reward_threshold,
            device
        )

        self.num_proposals = num_proposals
        self.enable_action_damping = enable_action_damping
        self.default_action_initialization = default_action_initialization
    

    # world model utilities

    def compute_latent_from_observation(self, current_obs_img):
        # apply torchvision transforms
        current_obs = self.world_model.transform(current_obs_img).unsqueeze(0).to(self.device)
        
        # encode the image
        z = self.world_model.encoder(current_obs)
        
        return z
    
    
    def compute_reward_from_latent(self, z, target_obj_embed):
        # predicted rewards for the current latent
        pred_reward, pred_bboxes = self.world_model.rewards_predictor(
            z,
            cond=target_obj_embed,
        )
        
        # sample from the distribution of the rewards
        pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
        pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
        pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                
        # predicted rewards for the current image
        return self.world_model.conf_pred_output_act(pred_reward)


    def apply_action_damping(self, action, pred_rewards):
        # apply a damping factor
        # rotation in axis-angle representation
        # rot_mat = R.from_euler('XYZ', action[:3]).as_matrix()
        action = action.squeeze(0)
        
        # rot_mat = py3d_transforms.euler_angles_to_matrix(action[3:], convention="XYZ")
        # rot_axang = py3d_transforms.matrix_to_axis_angle(rot_mat)
        # action[3:] = (1 / torch.exp(self.action_damping_factor["rotation"] * pred_rewards)) * rot_axang
        
        # # convert to rotation matrix
        # rot_mat = py3d_transforms.axis_angle_to_matrix(action[3:])
        
        # # convert to Euler Angles
        # action[3:] = py3d_transforms.matrix_to_euler_angles(rot_mat, convention="XYZ")
        
        # # translation
        # action[:3] = 1 / torch.exp(self.action_damping_factor["translation"] * pred_rewards) * action[:3]
                
        return action
    
    # TODO: Refactir
    def evaluate_rewards_for_actions(self, z, target_obj_embed, action):
        # action proposals
        action = torch.tensor(action, dtype=torch.float32).to(self.device)  # (horizon_len, 6)
            
        if action.ndim != 2 or action.shape[1] != 6:
            raise ValueError("Provided action list must have shape (horizon_len, 6)")
        
        horizon_len = action.shape[0]
        action = action.unsqueeze(1)  # (horizon_len, 1, 6)
        # action = init_action.detach().clone()
    
        # optimized actions
        action_list = action.detach()
        
        # initial latent state
        z_init = z.detach().clone()
        
        z = z_init.clone()
        predicted_znexts = []
        pred_rewards = []
        pred_bboxes = [] 
        pred_reward_stds = []
        
        for act in action_list:
            # initialize the latent state history
            z_hist = deque([z] * (self.world_model.latent_state_history_length - 1))
                
            z_curr_with_history = torch.cat(
                (z, *z_hist),
                dim=-2,
            )
            
            z_next = self.world_model.dynamics_predictor(z_curr_with_history, act.unsqueeze(0))[..., :z.shape[-2], :]
            
            z_next_mu, z_next_logvar = z_next[..., :self.world_model.dynamics_predictor.input_output_embed_dim], z_next[..., self.world_model.dynamics_predictor.input_output_embed_dim:]
            
            # sample from the distribution of the next latent state
            z_next_std = torch.exp(z_next_logvar / 2.0)
            predicted_znext = z_next_mu + torch.randn_like(z_next_std) * z_next_std
            
            predicted_znexts.append(predicted_znext)
            
            # rewards predictor
            pred_reward, pred_bboxes = self.world_model.rewards_predictor(predicted_znext, cond=target_obj_embed)
            
            # sample from the distribution of the rewards
            pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
            pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
            pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                    
            pred_reward = self.world_model.conf_pred_output_act(pred_reward)
            
            pred_rewards.append(pred_reward.item())
            pred_reward_stds.append(pred_reward_std.item())
            
        if pred_bboxes is not None:
            # apply the sigmoid function to the predicted bounding boxes
            pred_bboxes = self.world_model.sigmoid(pred_bboxes)
                        
            # apply the sigmoid function to the predicted bounding boxes
            pred_bboxes = pred_bboxes.squeeze()
            pred_bboxes.append(pred_bboxes)
            
        return action_list, predicted_znexts, pred_rewards, pred_bboxes, pred_reward_stds
    
    def get_actions(self, current_obs_img, target_obj_embed=None, curr_pose=None):
        
        z = self.compute_latent_from_observation(current_obs_img)
        
        # get predicted reward from latent
        curr_reward_predicted = self.compute_reward_from_latent(z, target_obj_embed)
        
        # decide controller
        use_gradient_planner = True
        action_proposals = None
        
        if curr_reward_predicted <= self.reinitialize_action_reward_threshold:
            # get next action proposals from action_generator
            action_proposals = self.action_generator.generate_action_proposals(
                current_obs_img, curr_pose, self.num_proposals
            )
            
            if not ((action_proposals is None) or (len(action_proposals) == 0)):
                use_gradient_planner = False
            
        if not use_gradient_planner:
            def get_best_plan(action_proposals):
                max_reward = -1
                max_idx = None
                all_plans = []
                all_cumulative_rewards = []

                for i, pred_action_sequence in enumerate(action_proposals):
                    
                    if not isinstance(pred_action_sequence[0], list):
                        pred_action_sequence = [pred_action_sequence]
                    
                    # get interpolated action sequence
                    action_sequence = self.interpolate_action_sequence(pred_action_sequence)
                    
                    if isinstance(self.action_generator, CEMActionGenerator):
                        # optimize action sequence
                        plan = self.evaluate_rewards_for_actions(
                            z=z,
                            target_obj_embed=target_obj_embed,
                            action=action_sequence,
                        )
                    else:
                        # optimize action sequence
                        plan = self.world_model.joint_gradient_planner(
                            z,
                            steps=20,
                            target_obj_embed=target_obj_embed,
                            verbose_print=False,
                            init_action=action_sequence,
                            return_all_sequence=True,
                        )
                    
                    all_plans.append(plan)
                    
                    predicted_reward_sequence = plan[2]
                
                    # TODO alternatives?
                    # cumulative_reward = sum(predicted_reward_sequence) / len(predicted_reward_sequence)
                    cumulative_reward = sum((i + 1) * x for i, x in enumerate(predicted_reward_sequence)) / len(predicted_reward_sequence)

                    # append to list
                    all_cumulative_rewards.append(cumulative_reward)

                    if cumulative_reward > max_reward:
                        max_reward = cumulative_reward
                        max_idx = i
                        
                return max_idx, all_plans, all_cumulative_rewards
                
            if isinstance(self.action_generator, CEMActionGenerator):
                for eval_idx in range(self.action_generator.num_evals):
                    max_idx, all_plans, all_cumulative_rewards = get_best_plan(
                        action_proposals=action_proposals,
                    )
                    
                    # update the action proposals
                    action_proposals = self.action_generator.generate_action_proposals(
                        current_obs_img, curr_pose, self.num_proposals,
                        samples=action_proposals,
                        rewards=all_cumulative_rewards,
                    )
                    
            else:
                max_idx, all_plans, _ = get_best_plan(
                    action_proposals=action_proposals,
                )
                
            # find the best plan
            best_plan = all_plans[max_idx]        
            print("plan by action init: ", best_plan[0])

        else:
            
            # default next action from gradient planner
            best_plan = self.world_model.joint_gradient_planner(
                    z,
                    horizon_len=1, # TODO default planner currently only plan for 1 step
                    target_obj_embed=target_obj_embed,
                    action_initialization=self.default_action_initialization,
                    verbose_print=False,
                )
            print("plan by grad plan: ", best_plan[0])

        # extract the relevant results from the planner
        pred_action, pred_next_z, next_reward_predicted, next_predicted_bbox, next_reward_pred_std = best_plan
        
        if self.enable_action_damping:
            # apply a damping factor
            # damp each element of pred_action
            for i in range(len(pred_action)):
                pred_action[i] = self.apply_action_damping(
                    action=pred_action[i],
                    pred_rewards=next_reward_predicted[i]
                )
            # pred_action = self.apply_action_damping(
            #     action=pred_action,
            #     pred_rewards=curr_reward_predicted
            # )
            
        return pred_action, next_reward_predicted, next_reward_pred_std
    