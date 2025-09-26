from PIL import Image
import numpy as np
import torch
import openai
import base64
import json
import ast
import io
import re
from pathlib import Path

from src.tests.planner import Planner

with open("openai_key.json", "r") as file:
    config = json.load(file)
    openai_api_key = config.get("openai_api_key")
    
client = openai.OpenAI(api_key=openai_api_key)

class VLMPlanner(Planner):
    def __init__(self, 
                 world_model,
                #  target, 
                 mode="single_step", 
                 k=10, 
                 step_size_limit=[0.02, 0.06]): # single_step, top-k
        super().__init__()
        self.world_model = world_model
        self.device = device
        # self.target = target
        self.k = k # number of action proposals
        self.min_step_size = step_size_limit[0] # minimum step size
        self.max_step_size = step_size_limit[1] # maximum step size
        
        self.mode = mode
        # self.initialize_conversation(target)
        
        
    def compose_singlestep_messages(self):
        system_message = {
            "role": "system",
            "content": f"""\
                You are a robot facing a table with some objects on it.\
                You can move around but you can't touch the objects.\
                The target object {self.target} is somewhere on this table.\
                Your goal is to get an observation with a top-down view of the target ({self.target}) and make it in the center of the frame. \
                You need to output an action that moves the camera to a desired pose where this goal would be achieved. \
                The action is represented as a translation vector [x, y, z] of the delta camera location (in units of meters), \
                and a rotation vector [roll, pitch, yaw] specifying the rotation of the camera in x/y/z axes respectively (in units of radians).\
                Try adjusting roll, pitch, yaw to get a top-down view of the target.\
                The coordinate frame is aligned such that +x is rightwards, +y is forwards, and +z is upwards (in the camera frame).\
                For your reference, the size of the table is around 1m * 2m. \
                The camera pose that you output should be relative to the current camera pose.\
            """,
        }

        def get_query_text(pose=None):

            query_text = f"""This is your current observation, please carefully analyze it and think step by step. \
                    If you think the current observation only contains the {self.target} and you can't get any closer, please output the string 'DONE'. \
                    Otherwise, please output the top {self.k} actions that you think would help you achieve the goal given the current observation. \
                    Each action is represented as a 6D array, where the first 3 elements represent camera translation [x, y, z], and the last 3 elements represent camera rotation [roll, pitch, yaw]. \
                    The coordinate frame is aligned such that +x is rightwards, +y is forwards, and +z is upwards (in the camera frame).\
                    Roll is rotation around the x-axis that allows you to tilt the camera up and down. \
                    Pitch is rotation around the y-axis that allows you to pan the camera left and right. \
                    Please limit your actions to a maximum of 0.3m for translation and 0.5 radians for rotation. \
                    Please think carefully step by step and reason about why the action you output can help you get the desired observation. \
                    Output the results in a structured JSON format as follows: \
                    {{ \
                        'actions': [ \
                            {{'rank': 1, 'confidence': <confidence_score>, 'action': [x1, y1, z1, roll1, pitch1, yaw1], 'explanation': <explanation1>}}, \
                            ... \
                            {{'rank': {self.k}, 'confidence': <confidence_score>, 'action': [x{self.k}, y{self.k}, z{self.k}, roll{self.k}, pitch{self.k}, yaw{self.k}], 'explanation': <explanation{self.k}>}}, \
                        ] \
                    }} \
                    Ensure the confidence scores are in descending order. \
                    Do not include any extra explanation outside of the JSON structure. \
                """
            return query_text
        return system_message, get_query_text
    
    def compose_multistep_messages(self):
      
#       Your goal is to get an observation with a top-down view of the target ({self.target}) and make it in the center of the frame. \
#       You need to output an action that moves the camera to a desired pose where this goal would be achieved. \
        system_message = {
            "role": "system",
            "content": f"""\
                You are a robot facing a table with some objects on it.\
                You can move around but you can't touch the objects.\
                Don't get too close to the table to avoid collisions, and don't get too far away to avoid losing sight of the target.\
                The target object {self.target} is somewhere on this table.\
                Your goal is to m0ve the camera as close to the target ({self.target}) as possible. \
                You need to output a action that moves the camera to a desired pose where this goal would be achieved. \
                The action is represented as a 6 dimensional vector, where the first 3 elements represent the translation (in units of meters), \
                and the last 3 elements represent the rotation (in units of radians).\
                Try adjusting the position and orientation of the camera to get a good top-down view of the target.\
                For your reference, the size of the table is around 1m * 2m. \
                The camera pose that you output should be relative to the current camera pose.\
            """,
        }

# <<<<<<< master
        query_text = f"""This is your current observation, please carefully analyze it and think step by step. \
                If you think the current observation only contains the {self.target} and you can't get any closer, please output the string 'DONE'. \
                Otherwise, please output the top {self.k} actions that you think would help you achieve the goal given the current observation. \
                Here are instructions on how to compose your output: \
                Your action should be represented as a 6d array, each element is defined as follows: \
                    - first element: translation to the right (a positive value) or to the left (a negative value) \
                    - second element: translation to the front (a positive value) or to the back (a negative value) \
                    - third element: translation upwards (a positive value) or downwards (a negative value) \
                    - fourth element: tilting the camera up (a positive value) or down (a negative value) \
                    - fifth element: rotating the camera image clockwise (a positive value) or counterclockwise (a negative value) \
                    - sixth element: panning the camera to the left (a positive value) or to the right (a negative value) \
                Please limit your actions to a maximum of 0.2m for translation and 0.5 radians for rotation. \
                Please think carefully step by step and reason about why the action you output can help you get the desired observation. \
                Output the results in a structured JSON format as follows: \
                {{ \
                    'actions': [ \
                        {{'rank': 1, 'confidence': <confidence_score>, 'action': [a1, b1, c1, d1, e1, 0], 'explanation': <explanation1>}}, \
                        ... \
                        {{'rank': {self.k}, 'confidence': <confidence_score>, 'action': [a{self.k}, b{self.k}, c{self.k}, d{self.k}, e{self.k}, 0], 'explanation': <explanation{self.k}>}}, \
                    ] \
                }} \
                Ensure the confidence scores are in descending order. \
                Do not include any extra explanation outside of the JSON structure. \
            """
            
        # query_text = f"""This is your current observation, please carefully analyze it and think step by step. \
        #         If you think the current observation only contains the {self.target} and you can't get any closer, please output the string 'DONE'. \
        #         Otherwise, please output the top {self.k} actions that you think would help you achieve the goal given the current observation. \
        #         Here are instructions on how to compose your output: \
        #         Your action should be represented as a 6d array, each element is defined as follows: \
        #             - first element: translation to the right (a positive value) or to the left (a negative value) \
        #             - second element: translation to the front (a positive value) or to the back (a negative value) \
        #             - third element: translation upwards (a positive value) or downwards (a negative value) \
        #             - fourth element: tilting the camera up (a positive value) or down (a negative value) \
        #             - fifth element: panning the camera to the left (a positive value) or to the right (a negative value) \
        #             - sixth element: rotating the camera counterclockwise (a positive value) or clockwise (a negative value) \
        #         Please limit your actions to a maximum of 0.3m for translation and 0.5 radians for rotation. \
        #         Please think carefully step by step and reason about why the action you output can help you get the desired observation. \
        #         Output the results in a structured JSON format as follows: \
        #         {{ \
        #             'actions': [ \
        #                 {{'rank': 1, 'confidence': <confidence_score>, 'action': [a1, b1, c1, d1, e1, 0], 'explanation': <explanation1>}}, \
        #                 ... \
        #                 {{'rank': {self.k}, 'confidence': <confidence_score>, 'action': [a{self.k}, b{self.k}, c{self.k}, d{self.k}, e{self.k}, 0], 'explanation': <explanation{self.k}>}}, \
        #             ] \
        #         }} \
        #         Ensure the confidence scores are in descending order. \
        #         Do not include any extra explanation outside of the JSON structure. \
        #     """
        return system_message, query_text
# =======
        def get_query_text(pose=None):

            query_text = f"""This is your current observation. Please carefully analyze the current observation and think about what is the best action to take givne what you see. \
                    If you think the current observation only contains the {self.target} and you can't get any closer, please output the string 'DONE'. \
                    Otherwise, please output the top {self.k} actions that you think would help you achieve the goal given the current observation. \
                    Here are instructions on how to compose your output: \
                    Your action should be represented as a 6d array, each element is defined as follows: \
                        - first element: translation to the right (a positive value) or to the left (a negative value) \
                        - second element: translation upwards (a positive value) or downwards (a negative value) \
                        - third element: translation backwards (a positive value) or forwards (a negative value) \
                        - fourth element: tilting the camera up (a positive value) or down (a negative value) \
                        - fifth element: panning the camera to the left (a positive value) or to the right (a negative value) \
                        - sixth element: rotating the camera counterclockwise (a positive value) or clockwise (a negative value) \
                    for example, if you want to move slightly right and then forward, then look down on the table, your action could be something like \
                    [0.1, 0, -0.2, -0.1, 0, 0] \
                    """ + self.describe_pose(pose) + f"""
                    Please limit your actions to a maximum of 0.2m for translation and 0.6 radians for rotation. \
                    This camera action that you output should be relative to the current camera pose.\
                    Please think carefully step by step and reason about why the action you output can help you get the desired observation. \
                    Output the results in a structured JSON format as follows: \
                    {{ \
                        'actions': [ \
                            {{'rank': 1, 'confidence': <confidence_score>, 'action': [a1, b1, c1, d1, e1, f1], 'explanation': <explanation1>}}, \
                            ... \
                            {{'rank': {self.k}, 'confidence': <confidence_score>, 'action': [a{self.k}, b{self.k}, c{self.k}, d{self.k}, e{self.k}, 0], 'explanation': <explanation{self.k}>}}, \
                        ] \
                    }} \
                    Ensure the confidence scores are in descending order. \
                    Do not include any extra explanation outside of the JSON structure. \
                """
            return query_text
        return system_message, get_query_text
# >>>>>>> vlm-prompting
    
    def initialize_conversation(self, target):
        self.target = target
        
        if self.mode == "single_step":
            self.system_message, self.get_query_text = self.compose_singlestep_messages()
        elif self.mode == "multi_step":
            self.system_message, self.get_query_text = self.compose_multistep_messages()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        self.conversation_history = [self.system_message]
        
        
    def encode_image_tensor(self, img_tensor):
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # encode the PIL image
        return self.encode_image_pil(img_pil)
        
        # TODO: Remove
        # buffered = io.BytesIO()
        # img_pil.save(buffered, format="JPEG")
        # img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # return img_base64
    
    def encode_image_pil(self, img_pil):
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64

    def mask_action(self, a):
        # if self.mode == "multi_step":
        #     a[2] = -a[2]
        return a

    def parse_actions_from_response(self, response_text):
        cleaned_text = response_text.strip("```json").strip("```")
        parsed_response = json.loads(cleaned_text)["actions"]
        
        # TODO when parsed_response is empty

        # extract the actions
        actions = []
        for a in parsed_response:
            actions.append(a["action"])
            
        return actions
    
    # TODO is it necessary to interpolate actions at random step sizes? If not, 
    # why do we enforce varying step sizes at training time?
    def interpolate_action(self, action, min_step_size=0.03, max_step_size=0.06):
        """
        Interpolate single-step action into smaller step sizes.
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

    def describe_pose(self, pose=None):
        if pose is None:
            return ""
        
        [x, y, z] = pose
        if self.mode == "multi_step":
            description = f"For your information, the center of the table is at (0, 0, 0), and the current camera pose is at ({x:.2f}, {y:.2f}, {z:.2f})."
        return description

    def query_for_action(self, obs, pose=None):

        encoded_obs = self.encode_image_pil(obs)
        self.conversation_history.append(
            {"role": "user", "content": [
                {"type": "input_text", "text": self.get_query_text(pose)},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded_obs}"}
            ]}
        )
        response = client.responses.create(
            model="gpt-4o",
            input=self.conversation_history
        )

        response_text = response.output[0].content[0].text
        print("response_text")
        print(response_text)

        self.conversation_history.append({"role": "assistant", "content": response_text})

        if response_text == "DONE":
            return None
        else:
            try:
                actions = self.parse_actions_from_response(response_text)
                return actions
            
            except Exception as e:
                print(e)
                return None
            
    def get_action(self, obs, target_obj_embed=None, curr_pose=None):
        
        action_proposals = self.query_for_action(obs, curr_pose)
        if action_proposals is None or len(action_proposals) == 1:
            print("Finished")
            return None, None, None
        
        max_idx = max_reward = 0
        best_actions = None
        best_rewards = None
        
        for i, pred_action in enumerate(action_proposals):
            start_pose = curr_pose

            # interpolate wm actions
            interpolated_actions = self.interpolate_action(pred_action)
            
            # initial obs from wm
            current_obs = self.world_model.transform(obs).unsqueeze(0).to(self.device)
            z = self.world_model.encoder(current_obs)
            cumulative_reward = 0
            
            tmp_rewards = []
                
            # roll out
            for j, action in enumerate(interpolated_actions):
                # predicted reward from current latent
                pred_reward, pred_bboxes = self.world_model.rewards_predictor(z, cond=target_obj_embed)
                pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
                pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
                raw_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std

                predicted_reward = self.world_model.conf_pred_output_act(raw_reward).item()

                tmp_rewards.append(predicted_reward)
                
                cumulative_reward += predicted_reward
                # apply action to latent
                action_tensor = torch.tensor(action).to(self.device)
                # z = self.world_model.dynamics_predictor(z, action_tensor.unsqueeze(0).unsqueeze(0))
                z = self.world_model.dynamics_predictor(
                    z, 
                    action_tensor.unsqueeze(0).unsqueeze(0),
                )[..., :z.shape[-2], :]

                z_mu, z_logvar = z[..., :self.world_model.dynamics_predictor.input_output_embed_dim], z[..., self.world_model.dynamics_predictor.input_output_embed_dim:]

                # sample from the distribution of the next latent state
                z_std = torch.exp(z_logvar / 2.0)
                z = z_mu + torch.randn_like(z_std) * z_std

            # NOTE averaged
            if len(interpolated_actions) > 0:
                cumulative_reward /= len(interpolated_actions)

            if cumulative_reward > max_reward:
                max_reward = cumulative_reward
                max_idx = i
                best_actions = interpolated_actions
                best_rewards = tmp_rewards
                
            print(f"Action {i}, Cumulative Reward: {cumulative_reward:.2f}")
        print(f"Best Action: {max_idx}, Cumulative Reward: {max_reward:.2f}")

        # TODO check data types
        return best_actions, best_rewards, best_rewards, "N/A" # TODO doesn't have next-pred-reward nor its std-dev
 
    def get_conversation_history(self):
        return self.conversation_history