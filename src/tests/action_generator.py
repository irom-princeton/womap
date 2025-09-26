from abc import ABC, abstractmethod
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

with open("openai_key.json", "r") as file:
    config = json.load(file)
    openai_api_key = config.get("openai_api_key")
    
client = openai.OpenAI(api_key=openai_api_key)

class ActionGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_action_proposals(self, obs, pose, num_proposals):
        """
        Returns:
            action_props: List of next action proposals, where each action proposal is a list of actions
        """
        pass

class GridActionGenerator(ActionGenerator):
    def __init__(self, 
                 step_size_mean=0.1,
                 step_size_std=0.05,
                 rotation_step_size_std=0.1):
        super().__init__()
        self.step_size_mean = step_size_mean
        self.step_size_std = step_size_std
        self.rotation_step_size_std = rotation_step_size_std

    def generate_action_proposals(self, obs, pose, num_proposals):
        # obs and pose are not used
        proposed_actions = []
        
        for _ in range(num_proposals):
            # sample action step sizes
            step_size = np.random.normal(self.step_size_mean, self.step_size_std)
            
            # sample direction in roll, pitch, yaw
            roll = np.random.normal(0, self.rotation_step_size_std)
            pitch = np.random.normal(0, self.rotation_step_size_std)
            yaw = np.random.normal(0, self.rotation_step_size_std)
            
            # create action with length step_size in the direction of roll, pitch, yaw
            action = [step_size * np.cos(roll), step_size * np.sin(pitch), step_size * np.sin(yaw),
                     roll, pitch, yaw]
            proposed_actions.append(action)
        return proposed_actions

class CEMActionGenerator(ActionGenerator):
    def __init__(self, 
                 dist_mean=0.05,
                 dist_std=0.05,
                 topk=2,
                 num_evals=5, # number of evals to update the mean and std
                ):
        super().__init__()
        self.dist_mean = dist_mean
        self.dist_std = dist_std
        self.topk = topk
        self.num_evals = num_evals
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_action_proposals(self, obs, pose, num_proposals, samples=None, rewards=None):
        # initialize the mean
        if samples is None or rewards is None:
            samp_mean = self.dist_mean * torch.ones(6, device=self.device)
            samp_std = self.dist_std * torch.ones(6, device=self.device)
        else:
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.device)
            
            if not isinstance(samples, torch.Tensor):
                samples = torch.tensor(samples).to(self.device)
            
            # get the Top-K samples
            topk_samples = samples[torch.argsort(rewards, descending=True)[:self.topk]]
            
            # compute from mean and standard deviation of the samples
            samp_mean = torch.mean(topk_samples, dim=0, keepdim=False)
            samp_std = torch.std(topk_samples, dim=0, keepdim=False)
        
        # obs and pose are not used
        # sample num_proposals actions
        proposed_actions = [
            (samp_mean + samp_std * torch.randn(6).to(self.device)).detach().cpu().numpy().tolist()
            for _ in range(num_proposals)
        ]
        
        return proposed_actions
       
class StepActionGenerator(ActionGenerator):
    def __init__(self, query_mode):
        super().__init__()
        self.query_mode = query_mode
        
    def generate_action_proposals(self, obs, pose, num_proposals):
        if self.query_mode == "atomic-base":
            actions = [
                [0, 0, -0.15, 0, 0, 0],            # A
                [-0.15, 0, 0, 0, 0, 0],            # B
                [0.15, 0, 0, 0, 0, 0],             # C
                [0, 0, 0, 0, np.pi/4, 0],          # D
                [0, 0, 0, 0, -np.pi/4, 0],         # E
                [-0.15, 0, -0.15, 0, -np.pi/3, 0], # F
                [0.15, 0, -0.15, 0, np.pi/3, 0]    # G
            ]

        elif self.query_mode == "test-v1":
            actions = [
                [0, 0, -0.05, 0, 0, 0],                  # A
                [0, 0, -0.15, 0, 0, 0],                  # B
                [0, 0, -0.30, 0, 0, 0],                  # C
                [-0.15, 0, 0, 0, 0, 0],                  # D
                [0.15, 0, 0, 0, 0, 0],                   # E
                [0, 0, 0, 0, np.pi/4, 0],                # F
                [0, 0, 0, 0, -np.pi/4, 0],               # G
                [-0.15, 0, -0.15, 0, -np.pi/3, 0],       # H
                [0.15, 0, -0.15, 0, np.pi/3, 0]          # I
            ]
        else:
            raise ValueError(f"Invalid mode: {self.query_mode}")
        return actions
    
class TopdownActionGenerator(ActionGenerator):
    def __init__(self):
        super().__init__()
        
    def generate_action_proposals(self, obs, pose, num_proposals):
        raise NotImplementedError("TopdownActionGenerator is not implemented yet.")
    
class VLMActionGenerator(ActionGenerator):
    def __init__(self, query_mode="", num_proposals=4):
        super().__init__()
        self.query_mode = query_mode
        self.k = num_proposals

    def compose_message_atomic_base(self, pose=None):
        system_message = {
            "role": "system",
            "content": f"""\
                You are a robot facing a table with some objects on it.\
                You can move around but you can't touch the objects.\
                Don't get too close to the table surface to avoid collisions, and don't get too far away to avoid losing sight of the target.\
                The target object {self.target} is somewhere on this table.\
                Your goal is to get a good observation of the {self.target} by move the camera as close to the target as possible. \
                At each round, we will provide you with the current camera observation, and several actions that you can take. \
                You will need to choose from the 
                For your reference, the size of the table is around 1m * 2m. \
                The camera pose that you output should be relative to the current camera pose.\
            """,
        }
        
        def get_query_text(pose=None):
            query_text = f"""This is your current observation. Please carefully analyze the current observation and think about what is the best action to take given what you see. \
                    If you think the current observation only contains the {self.target} and you can't get any closer, please output the string 'DONE'. \
                    Otherwise, please select the top {self.k} choices from the action options below that you think would help you achieve the goal given the current observation. \
                    The options are: \
                        (A) Move camera directly forward for 15 cm \
                        (B) Transition camera to the left for 15 cm \
                        (C) Transition camera to the right for 15 cm \
                        (D) Pan the camera to the left for 45 degrees \
                        (E) Pan the camera to the right for 45 degrees \
                        (F) Move the camera in the forward-left direction and then look to the right \
                        (G) Move the camera in the forward-right direction and then look to the left \
                    All these actions should be executed relative to the current camera position.\
                    Please think carefully step by step and reason about why your choice can help you get the desired observation. \
                    Output the results in a structured JSON format as follows: \
                    {{ \
                        "descriptions": <what you observe in the the current scene and whether there are hints of the {self.target}.>
                        "actions": [ \
                            {{"rank": 1, "choice": <choice1>, "confidence": <confidence_score1>, "explanation": <explanation1>}}, \
                            ... \
                            {{"rank": {self.k}, "choice": <choice{self.k}>, "confidence": <confidence_score{self.k}>, "explanation": <explanation{self.k}>}}, \
                        ] \
                    }} \
                    <choice1>, ... <choice{self.k}> should be a singke letter representing one of the 7 choices above. \
                    Ensure the confidence scores are in descending order. \
                    Do not include any extra explanation outside of the JSON structure. \
                """
            return query_text
        return system_message, get_query_text
    
    def compose_message_test_v1(self, pose=None):
        system_message = {
            "role": "system",
            "content": f"""\
                You are looking at a table with some objects on it.\
                You can move around but you can't touch the objects.\
                Don't get too close to the table surface or you would collide, and don't get too far away or you will lose sight of the target.\
                The target object {self.target} is somewhere on this table.\
                Your goal is to find it by getting a good observation of the {self.target} and get as close to it as possible. \
                Looking at the table, you can choose to go forward, left, right. \
                You can also look to the left or right. \
                There are options combining these actions too. \
                You will need to choose from the list of these actions. \
                For your reference, the size of the table is around 1m * 2m. \
                The action you choose should be relative to your current situation.\
            """,
        }
        
        def get_query_text(pose=None):
            query_text = f"""This is what you currently see. Please carefully analyze the current observation and think about what is the best action to take given what you see. \
                    If you think the current observation is a good enough view of {self.target} and you can't get any closer, please say 'DONE'. \
                    Otherwise, please select the top {self.k} choices from the action options below that you think would help you achieve the goal given the current observation. \
                    The options are: \
                        (A) Move directly forward for 15 cm -- this lets you approach the objects in view\
                        (B) Move directly to the left for 15 cm -- this expands your left left by a bit\
                        (C) Move directly to the right for 15 cm -- this expands your right left by a bit\
                        (D) Look to the left by 45 degrees -- this lets you look around\
                        (E) Look to the right for 45 degrees -- this lets you look around\
                        (F) Move forward-left for 15 cm -- this lets you approach objects on the left side of your view\
                        (G) Move forward-right for 15 cm -- this lets you approach objects on the right side of your view\
                        (H) Move forward-left for 15 cm and then look right by 45 degrees-- this lets you look behind an object\
                        (I) Move forward-right for 15 cm and then look left by 45 degrees-- this lets you look behind an object\
                    All these actions should be executed relative to the current your position.\
                    Please think carefully step by step and reason about why your choice can help you get the desired observation. \
                    Please also review your movement history to see where you've already explored. \
                    Output the results in a structured JSON format as follows: \
                    {{ \
                        "descriptions": <what you observe in the the current scene and whether there are hints of the {self.target}.>
                        "actions": [ \
                            {{"rank": 1, "choice": <choice1>, "confidence": <confidence_score1>, "explanation": <explanation1>}}, \
                            ... \
                            {{"rank": {self.k}, "choice": <choice{self.k}>, "confidence": <confidence_score{self.k}>, "explanation": <explanation{self.k}>}}, \
                        ] \
                    }} \
                    <choice1>, ... <choice{self.k}> should be a singke letter representing one of the 7 choices above. \
                    Ensure the confidence scores are in descending order. \
                    Do not include any extra explanation outside of the JSON structure. \
                """
            return query_text
        return system_message, get_query_text
    
    def compose_message_atomic_v2(self, pose=None):
        system_message = {
            "role": "system",
            "content": f"""\
                You are a robot facing a table with some objects on it.\
                You can move around but you can't touch the objects.\
                Don't get too close to the table surface to avoid collisions, and don't get too far away to avoid losing sight of the target.\
                The target object {self.target} is somewhere on this table.\
                Your goal is to get a good observation of the {self.target} by move the camera as close to the target as possible. \
                At each round, we will provide you with the current camera observation, and several actions that you can take. \
                You will need to choose from the 
                For your reference, the size of the table is around 1m * 0.6m. \
                The camera pose that you output should be relative to the current camera pose.\
            """,
        }
        
        def get_query_text(pose=None):
            query_text = f"""This is your current observation. Please carefully analyze the current observation and think about what is the best action to take given what you see. \
                    If you think the current observation only contains the {self.target} and you can't get any closer, please output the string 'DONE'. \
                    Otherwise, please select the top {self.k} choices from the action options below that you think would help you achieve the goal given the current observation. \
                    The options are: \
                        (A) Move camera directly forward for 5 cm \
                        (B) Move camera directly forward for 15 cm \
                        (C) Move camera directly forward for 30 cm \
                        (D) Transition camera to the left for 15 cm \
                        (E) Transition camera to the right for 15 cm \
                        (F) Pan the camera to the left for 45 degrees \
                        (G) Pan the camera to the right for 45 degrees \
                        (H) Move the camera in the forward-left direction and then look to the right \
                        (I) Move the camera in the forward-right direction and then look to the left \
                    All these actions should be executed relative to the current camera position.\
                    Please think carefully step by step and reason about why your choice can help you get the desired observation. \
                    Output the results in a structured JSON format as follows: \
                    {{ \
                        "descriptions": <what you observe in the the current scene and whether there are hints of the {self.target}.>
                        "actions": [ \
                            {{"rank": 1, "choice": <choice1>, "confidence": <confidence_score1>, "explanation": <explanation1>}}, \
                            ... \
                            {{"rank": {self.k}, "choice": <choice{self.k}>, "confidence": <confidence_score{self.k}>, "explanation": <explanation{self.k}>}}, \
                        ] \
                    }} \
                    <choice1>, ... <choice{self.k}> should be a singke letter representing one of the 7 choices above. \
                    Ensure the confidence scores are in descending order. \
                    Do not include any extra explanation outside of the JSON structure. \
                """
            return query_text
        return system_message, get_query_text

    def initialize_conversation(self, target, num_proposals=None): # TODO hard code K=4
        self.target = target
        if num_proposals is not None:
            self.k = num_proposals
        
        if self.query_mode == "atomic-base":
            self.system_message, self.get_query_text = self.compose_message_atomic_base()
        elif self.query_mode == "test-v1":
            self.system_message, self.get_query_text = self.compose_message_test_v1()
        elif self.query_mode == "atomic-v2":
            self.system_message, self.get_query_text = self.compose_message_atomic_v2()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        self.conversation_history = [self.system_message]
        self.response_history = []

    def parse_actions_from_response(self, response_text):
        cleaned_text = response_text.strip("```json").strip("```")
        
        self.response_history.append(cleaned_text)
        
        if cleaned_text == "DONE":
            return None
        
        parsed_response = json.loads(cleaned_text)["actions"]
        
        actions = []
        
        # extract the actions
        if self.query_mode == "atomic-base":
            for a in parsed_response:
                choice = a["choice"]
                if choice in {"A", "(A)", "a", "(a)"}:
                    actions.append([0, 0, -0.15, 0, 0, 0])
                elif choice in {"B", "(B)", "b", "(b)"}:
                    actions.append([-0.15, 0, 0, 0, 0, 0])
                elif choice in {"C", "(C)", "c", "(c)"}:
                    actions.append([0.15, 0, 0, 0, 0, 0])
                elif choice in {"D", "(D)", "d", "(d)"}:
                    actions.append([0, 0, 0, 0, np.pi/4, 0])
                elif choice in {"E", "(E)", "e", "(e)"}:
                    actions.append([0, 0, 0, 0, -np.pi/4, 0])
                elif choice in {"F", "(F)", "f", "(f)"}:
                    actions.append([-0.15, 0, -0.15, 0, -np.pi/3, 0])
                elif choice in {"G", "(G)", "g", "(g)"}:
                    actions.append([0.15, 0, -0.15, 0, np.pi/3, 0])
                elif choice in {"DONE", "(DONE)", "done", "(done)"}:
                    return None
                else:
                    raise ValueError(f"Invalid choice: {choice}")
        elif self.query_mode == "test-v1":
            for a in parsed_response:
                choice = a["choice"]
                if choice in {"A", "(A)", "a", "(a)"}:
                    actions.append([0, 0, -0.15, 0, 0, 0])
                elif choice in {"B", "(B)", "b", "(b)"}:
                    actions.append([-0.15, 0, 0, 0, 0, 0])
                elif choice in {"C", "(C)", "c", "(c)"}:
                    actions.append([0.15, 0, 0, 0, 0, 0])
                elif choice in {"D", "(D)", "d", "(d)"}:
                    actions.append([0, 0, 0, 0, np.pi/4, 0])
                elif choice in {"E", "(E)", "e", "(e)"}:
                    actions.append([0, 0, 0, 0, -np.pi/4, 0])
                elif choice in {"F", "(F)", "f", "(f)"}:
                    actions.append([-0.15, 0, -0.15, 0, 0, 0])
                elif choice in {"G", "(G)", "g", "(g)"}:
                    actions.append([0.15, 0, -0.15, 0, 0, 0])
                elif choice in {"H", "(H)", "h", "(h)"}:
                    actions.append([-0.15, 0, -0.15, 0, -np.pi/4, 0])
                elif choice in {"I", "(I)", "i", "(i)"}:
                    actions.append([0.15, 0, -0.15, 0, np.pi/4, 0])
                elif choice in {"DONE", "(DONE)", "done", "(done)"}:
                    return None
                else:
                    raise ValueError(f"Invalid choice: {choice}")
                
        elif self.query_mode == "atomic-v2":
            for a in parsed_response:
                choice = a["choice"]
                if choice in {"A", "(A)", "a", "(a)"}:
                    actions.append([0, 0, -0.05, 0, 0, 0])
                elif choice in {"B", "(B)", "b", "(b)"}:
                    actions.append([0, 0, -0.15, 0, 0, 0])
                elif choice in {"C", "(C)", "c", "(c)"}:
                    actions.append([0, 0, -0.30, 0, 0, 0])
                elif choice in {"D", "(D)", "d", "(d)"}:
                    actions.append([-0.15, 0, 0, 0, 0, 0])    
                elif choice in {"E", "(E)", "e", "(e)"}:
                    actions.append([0.15, 0, 0, 0, 0, 0])
                elif choice in {"F", "(F)", "f", "(f)"}:
                    actions.append([0, 0, 0, 0, np.pi/4, 0])
                elif choice in {"G", "(G)", "g", "(g)"}:
                    actions.append([0, 0, 0, 0, -np.pi/4, 0])
                elif choice in {"H", "(H)", "h", "(h)"}:
                    actions.append([-0.15, 0, -0.15, 0, -np.pi/3, 0])
                elif choice in {"I", "(I)", "i", "(i)"}:
                    actions.append([0.15, 0, -0.15, 0, np.pi/3, 0])
                elif choice in {"DONE", "(DONE)", "done", "(done)"}:
                    return None
                else:
                    raise ValueError(f"Invalid choice: {choice}")
                
        return actions
    
    def encode_image_pil(self, img_pil):
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64
    
    def encode_image_tensor(self, img_tensor):
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        return self.encode_image_pil(img_pil)


    def generate_action_proposals(self, obs, pose=None, num_proposals=None):
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