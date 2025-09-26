import numpy as np
from typing import Literal
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
from collections import deque

import gc
import os
import sys
import copy
from pathlib import Path
import cv2

from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.helper import init_opt
from src.transforms import make_transforms, make_inv_transforms
from src.utils.tensors import init_weights
from src.utils.loss_utils import cross_entropy_loss_fn, smooth_l1_loss_fn, ciou_loss_fn, kl_divergence_loss_fn
# from src.utils.summary_utils import print_number_of_model_parameters
from src.utils.wm_utils import print_number_of_model_parameters
from src.datasets.active_perception_dataset import ActivePerceptionDataset

sys.path.append(f"{Path(__file__).parent.parent}")

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# TODO does not support distributed training
# TODO validation not added

class WorldModel:

    def __init__(
            self,
            cfg,
            device,
            encoder, 
            dynamics_predictor,
            rewards_predictor,
            decoder=None,
            mode="inference",
            ablation=None,
            encoder_frozen=True,
            verbose_print: bool = False,
        ):
        # configuration
        self.cfg = cfg
        self.device = device
        self.encoder = encoder.to(device)
        self.target_encoder = copy.deepcopy(encoder)
        self.dynamics_predictor = dynamics_predictor.to(device)
        self.rewards_predictor = rewards_predictor.to(device)
        self.decoder = decoder.to(device)
        self.conf_pred_output_act = torch.nn.Sigmoid()
        self.sigmoid = torch.nn.Sigmoid()

        # important parameters
        self.num_epochs = self.cfg.training.epochs
        self.batch_size = self.cfg.training.batch_size
        self.img_resize_shape = self.cfg.shared.img_resize_shape
        self.transform = make_transforms(self.img_resize_shape)
        self.inv_transform = make_inv_transforms(self.img_resize_shape)
        self.sequence_length = self.cfg.training.sequence_length
            
        # latent state history length
        self.latent_state_history_length = self.cfg.training.latent_state_history_length

        self.mode = mode
        self.encoder_frozen = encoder_frozen
        self.ablation=ablation

        # initialize weights
        # dynamics predictor
        for m in self.dynamics_predictor.modules():
            init_weights(m)
        # rewards predictor
        for m in self.rewards_predictor.modules():
            init_weights(m)
        # decoder   
        if self.decoder is not None:
            for m in self.decoder.modules():
                init_weights(m)
        
        # train setup
        if mode == "train":
            self.train_setup()
        elif mode == "inference":
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.dynamics_predictor.parameters():
                p.requires_grad = False
            for p in self.rewards_predictor.parameters():
                p.requires_grad = False
            if self.decoder is not None:
                for p in self.decoder.parameters():
                    p.requires_grad = False
                
        print("WorldModel initialized:")
        if verbose_print:
            # print the number of parameters for each model
            # encoder
            print("encoder=", self.encoder)
            
            # dynamics predictor
            print("dynamics_predictor=", self.dynamics_predictor)
            
            # rewards predictor
            print("rewards_predictor=", self.rewards_predictor)
            
            # decoder
            if self.decoder is not None:
                print("decoder=", self.decoder)
                
        # print the number of parameters for each model
        # encoder
        print_number_of_model_parameters(self.encoder,
                                            title=f"encoder: {self.encoder.name}")
        
        if self.dynamics_predictor is not None:
            # dynamics predictor
            print_number_of_model_parameters(self.dynamics_predictor,
                                                title=f"dynamics_predictor: {self.dynamics_predictor.name}")
            
        if self.rewards_predictor is not None:
            # rewards predictor
            print_number_of_model_parameters(self.rewards_predictor,
                                                title=f"rewards_predictor: {self.rewards_predictor.name}")
        
        if self.decoder is not None:    
            # decoder
            if self.decoder is not None:
                print_number_of_model_parameters(decoder,
                                                    title=f"decoder: {self.decoder.name}")
        
    def train_setup(self):
        # ablations
        if self.ablation is not None and isinstance(self.ablation, str):
            if self.ablation.lower() == "dynamics":
                self.ablate_dynamics()
            elif self.ablation.lower() == "rewards":
                self.ablate_rewards()

        # initialize dataset and loader
        train_path = self.cfg.dataset.root_path + "/" + self.cfg.dataset.train_data
        self.train_dataset = ActivePerceptionDataset(
            root_path=train_path,
            transform=self.transform,
            sequence_length=self.sequence_length,
            consecutive_obs=self.cfg.training.consecutive_obs,
            confidence_bin=self.cfg.training.confidence_bin,
            use_weighted_rewards=getattr(self.cfg.dataset, "use_weighted_rewards", False),
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
        )

        # initialize optimizer, scheduler, and scaler
        self.ipe = len(self.train_loader)
        self.optimizer, self.scaler, self.scheduler, self.wd_scheduler = init_opt(
            encoder=self.encoder,
            predictor=self.dynamics_predictor,
            decoder=self.decoder,
            detection_confidence_predictor=self.rewards_predictor,
            freeze_encoder=self.encoder_frozen,
            iterations_per_epoch=self.ipe,
            start_lr=self.cfg.optimization.start_lr,
            ref_lr=self.cfg.optimization.lr,
            warmup=self.cfg.optimization.warmup,
            num_epochs=self.num_epochs,
            wd=self.cfg.optimization.weight_decay,
            final_wd=self.cfg.optimization.final_weight_decay,
            final_lr=self.cfg.optimization.final_lr,
            use_bfloat16=self.cfg.training.use_bfloat16,
            ipe_scale=self.cfg.optimization.ipe_scale,
        )
        
        # gradients
        if self.encoder_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # scheduler
        ipe_scale = self.cfg.optimization.ipe_scale
        ema = self.cfg.optimization.ema

        self.momentum_scheduler = (
            ema[0] + i * (ema[1] - ema[0]) / (self.ipe * self.num_epochs * ipe_scale)
            for i in range(int(self.ipe * self.num_epochs * ipe_scale) + 1)
        )

    def load_rewards_predictor_weights(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]
        pretrained_dict = checkpoint["rewards_predictor"]
        if self.mode != "train":
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        msg = self.rewards_predictor.load_state_dict(pretrained_dict)
        print(f"loaded rewards predictor from epoch {epoch} with msg: {msg}")

    def load_dynamics_predictor_weights(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]
        pretrained_dict = checkpoint["dynamics_predictor"]
        if self.mode != "train":
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        msg = self.dynamics_predictor.load_state_dict(pretrained_dict)
        print(f"loaded dynamics predictor from epoch {epoch} with msg: {msg}")

    def load_decoder_weights(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]
        pretrained_dict = checkpoint["decoder"]
        if self.mode != "train":
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        msg = self.decoder.load_state_dict(pretrained_dict)
        print(f"loaded decoder from epoch {epoch} with msg: {msg}")
    
    def save_checkpoint(self, epoch, models_dict=None): 
        # dict of models
        if models_dict is None:
            models_dict = {
                "dynamics_predictor": self.dynamics_predictor,
                "target_encoder": self.target_encoder,
                "decoder": self.decoder,
                "rewards_predictor": self.rewards_predictor,
            }
        
        if not self.encoder_frozen:
            models_dict["encoder"] = self.encoder

        save_dict = {
            "opt": self.optimizer.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "epoch": epoch,
            "loss": None, #self.loss_meter.avg,
            "batch_size": self.batch_size,
            "world_size": None, #self.world_size,
            "lr": self.cfg.optimization.lr,
        }

        for mod_name, mod in models_dict.items():
            if mod is not None:
                save_dict[mod_name] = mod.state_dict()
        
        torch.save(save_dict, self.latest_path)
        if (epoch + 1) % self.cfg.training.checkpoint_freq == 0:
            torch.save(save_dict, self.save_path.format(epoch=f"{epoch + 1}"))
            print("Saved checkpoint at epoch", epoch + 1, "to", self.save_path.format(epoch=f"{epoch + 1}"))
    
    def train(self, wandb_projname, wandb_expname, log_images=False):
        self.folder = f"{Path(__file__).parent.parent.parent}/{self.cfg.logging.folder}/{wandb_projname}/{wandb_expname}"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.save_path = os.path.join(self.folder, f"{wandb_expname}" + "-ep{epoch}.pth.tar")
        self.latest_path = os.path.join(self.folder, f"{wandb_expname}-latest.pth.tar")
        
        def loss_fn(z, h):
            return smooth_l1_loss_fn(z, h)
        
        def conf_loss_fn(pred, gt):
            return cross_entropy_loss_fn(pred, gt)
        
        def forward_target(next_img):
            with torch.no_grad():
                h = self.target_encoder(next_img)
                # TODO normalization?
                # h = F.layer_norm(
                #     h, (h.size(-1),)
                # ) 
                
                return h
        
        def bbox_loss_fn(pred, gt):
            # return mse_loss_fn(pred, gt)
            # loss = ciou_loss_fn(pred, gt)
            # loss = cross_entropy_loss_fn(pred, gt)
            loss = smooth_l1_loss_fn(pred, gt, beta=1e-4)
            return loss
        
        def decoder_loss_fn(pred, gt):
            # return mse_loss_fn(pred, gt)
            return smooth_l1_loss_fn(pred, gt, beta=1e-4)

        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            for i, sampled_batch in enumerate(tqdm(self.train_loader, desc=f"Epoch: {epoch}")):


                # TODO: Temporary fix for error in the length of the dataloader
                if i >= len(self.train_loader) - 1:
                    continue

                observation_sequence = sampled_batch["observation_sequence"].moveaxis(1, 0) # permute(1, 0, 2, 3, 4)
                actions = sampled_batch["actions"].moveaxis(1, 0) # permute(1, 0, 2)
                detection_confidence_scores = sampled_batch["detection_confidence_scores"].moveaxis(1, 0) # permute(1, 0)
                bboxes = sampled_batch["bboxes"].moveaxis(1, 0) # permute(1, 0, 2)
                
                # image dimension
                img_data_h_w = observation_sequence.shape[-2:]
                
                # normalize the bounding-boxes (assumes a square image)
                bboxes = bboxes / img_data_h_w[0]
                
                # text embeddings of the target object
                cond_embeds = torch.broadcast_to(
                    self.train_dataset.target_objects_embeds,
                    (
                        observation_sequence.shape[1],
                        *self.train_dataset.target_objects_embeds.shape,
                    ),
                )
                
                # randomly select target objects for supervision
                sel_target_obj_idx = torch.randint(low=0, high=detection_confidence_scores.shape[-1], size=(observation_sequence.shape[1],))

                # select the corresponding mini-batch
                detection_confidence_scores = torch.gather(
                    detection_confidence_scores,
                    dim=-1,
                    index=torch.broadcast_to(
                        sel_target_obj_idx[None, :, None],
                        (
                            *detection_confidence_scores.shape[:-1],
                            1,
                        ),
                    ),
                ).squeeze()
                bboxes = torch.gather(
                    bboxes,
                    dim=-2,
                    index=torch.broadcast_to(
                        sel_target_obj_idx[None, :, None, None],
                        (
                            *bboxes.shape[:-2],
                            1,
                            bboxes.shape[-1],
                        ),
                    ),
                ).squeeze()
                cond_embeds = torch.gather(
                    cond_embeds,
                    dim=-2,
                    index=torch.broadcast_to(
                        sel_target_obj_idx[:, None, None],
                        (
                            *cond_embeds.shape[:-2],
                            1,
                            cond_embeds.shape[-1],
                        ),
                    ),
                )
                
                cond_embeds = cond_embeds.to(self.device, non_blocking=True)
                
                def train_step():
                    self.optimizer.zero_grad()

                    # loss components
                    dynamics_loss = 0
                    detection_loss = 0
                    decoder_loss = 0
                    bbox_loss = 0

                    # torchvision transform Resize
                    tv_transforms_resize = None
                    
                    # initialize torchvision transform Resize
                    viz_tv_transforms_resize = tv_transforms.Resize(
                        img_data_h_w
                    )
                    
                    # weight for the rewards
                    rewards_loss_weight = 1.0 / 3.0
                    
                    # weight for the bounding-box loss
                    bbox_loss_weight = 1.0 / 3.0
                    
                    # weight for the KL-divergence
                    kl_divergence_weight = 2e-3

                    # decoded and ground-truth states
                    stacked_decoded_imgs = []
                        
                    # initial observation (timestep t0)
                    img_curr = observation_sequence[0].to(self.device, non_blocking=True)
                    
                    # detection confidence at t(0)
                    score_curr = detection_confidence_scores[0].to(self.device, non_blocking=True, dtype=img_curr.dtype)
                    
                    # bounding box at t(0)
                    bbox_curr = bboxes[0].to(self.device, non_blocking=True, dtype=img_curr.dtype)
                    
                    # TODO: Fix the repetition in this code section
                    # latent state
                    if self.encoder_frozen:
                        with torch.no_grad():
                            z_curr = self.encoder(img_curr).to(img_curr.dtype)
                    else:
                        z_curr = self.encoder(img_curr).to(img_curr.dtype)
                    
                    # # reshape, if necessary (for CLIP)
                    # if len(z_curr.shape) > 3:
                    #     z_curr = z_curr.reshape(
                    #         z_curr.shape[0],
                    #         z_curr.shape[-3] * z_curr.shape[-2],
                    #         z_curr.shape[-1],
                    #     )
                        
                    if self.rewards_predictor is not None:      
                        # predicted detection confidence
                        pred_det_conf, pred_bboxes = self.rewards_predictor(z_curr, cond=cond_embeds)
                        
                        # sample from the distribution of the rewards
                        pred_det_conf_mean, pred_det_conf_logvar = pred_det_conf[..., 0:1], pred_det_conf[..., 1:2]
                        pred_det_conf_std = torch.exp(pred_det_conf_logvar / 2.0)
                        pred_det_conf = pred_det_conf_mean + torch.randn_like(pred_det_conf_std) * pred_det_conf_std
                        
                        # compute the detection confidence loss
                        det_conf_loss = (
                            conf_loss_fn(
                                pred_det_conf.float(), score_curr[:, None].float()
                            )
                            + kl_divergence_weight * kl_divergence_loss_fn(
                                pred_det_conf_mean, pred_det_conf_logvar
                            )
                        )
                        detection_loss += det_conf_loss
                        
                        if pred_bboxes is not None:
                            # apply the sigmoid function to the predicted bounding boxes
                            pred_bboxes = self.sigmoid(pred_bboxes)
                    
                            # compute the loss for the predicted bounding boxes (assuming square images)
                            bbox_loss += bbox_loss_fn(
                                pred_bboxes.float(),
                                bbox_curr.float()
                            )
                            # bbox_loss += conf_loss_fn(pred_bboxes.float(), bbox_curr.float())
                            
                            # # apply the sigmoid function to the predicted bounding boxes
                            # pred_bboxes = self.conf_pred_output_act(pred_bboxes)
                    
                    # decode the latent states
                    if self.decoder is not None:
                        # Remove the gradients of the inputs from the computation graph
                        # so we can train the decoder and the encoder independently.
                        z_curr_decode, _ = self.decoder(z_curr.detach().unsqueeze(1))

                        # reshape
                        z_curr_decode = z_curr_decode.reshape(
                            img_curr.shape[0], -1, *z_curr_decode.shape[-3:]
                        ).squeeze()

                        if tv_transforms_resize is None:
                            # initialize torchvision transform Resize
                            tv_transforms_resize = tv_transforms.Resize(
                                z_curr_decode.shape[-2:]
                            )

                        # resize the target image for the decoder loss
                        resz_curr_img = tv_transforms_resize(img_curr)

                        # decoder loss
                        decoder_loss += decoder_loss_fn(z_curr_decode, resz_curr_img)

                        # decoded and ground-truth states
                        decoded_imgs = []

                        # random image index
                        rand_img_idx = torch.randint(
                            low=0,
                            high=resz_curr_img.shape[0],
                            size=(1,),
                        )

                        # resize the target image for visualization
                        viz_resz_curr_img = viz_tv_transforms_resize(img_curr)
                        
                        # ground-truth image
                        dec_bbox_curr = bbox_curr * img_data_h_w[0]
                        decoded_imgs.append(
                            wandb.Image(
                                data_or_path=viz_resz_curr_img[rand_img_idx]
                                .moveaxis(-3, -1)
                                .detach()
                                .cpu()
                                .numpy(),
                                caption="Encoder: GT image",
                                boxes={
                                    "predictions": {
                                        "box_data": [
                                            {
                                                "position": {
                                                    "minX": dec_bbox_curr[rand_img_idx, 0].item(),
                                                    "maxX": dec_bbox_curr[rand_img_idx, 2].item(),
                                                    "minY": dec_bbox_curr[rand_img_idx, 1].item(),
                                                    "maxY": dec_bbox_curr[rand_img_idx, 3].item(),
                                                },
                                                "domain": "pixel",
                                                "class_id": 1,
                                                "box_caption": f"{self.train_dataset.target_objects[sel_target_obj_idx[rand_img_idx]]}: {score_curr[rand_img_idx].item():.2f}", # TODO: Update this code.
                                                "scores": {"conf": score_curr[rand_img_idx].item()}
                                            }
                                        ],
                                        # "class_labels": None, # TODO: Update this code.
                                    }
                                }
                            )
                        )
                        
                        # # ground-truth image
                        # gt_img_rgb = (
                        #     viz_resz_curr_img[rand_img_idx]
                        #         .moveaxis(-3, -1)
                        #         .detach()
                        #         .cpu()
                        #         .numpy()
                        #         * 255
                        # ).squeeze().astype(np.uint8)
                        
                        # # annotate bounding-box
                        # # RGB-to-BGR
                        # cv2_img = cv2.cvtColor(gt_img_rgb, cv2.COLOR_RGB2BGR)
                        # # cv2_img = gt_img_rgb[..., ::-1]
                        # gt_img_bboxes = bbox_curr.detach().cpu().numpy()
                        # cv2.rectangle(
                        #     cv2_img,
                        #     (int(gt_img_bboxes[rand_img_idx][0]),
                        #      int(gt_img_bboxes[rand_img_idx][1])),
                        #     (int(gt_img_bboxes[rand_img_idx][2]),
                        #      int(gt_img_bboxes[rand_img_idx][3])),
                        #     (255, 0, 0),
                        #     2
                        # )
                        # # BGR-to-RGB
                        # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        # # cv2_img = cv2_img[..., ::-1]
                        # cv2_img = cv2_img.astype(float) / 255.0
                        
                        # decoded_imgs.append(
                        #     wandb.Image(
                        #         data_or_path=gt_img_rgb.astype(float) / 255.0, #cv2_img[None],
                        #         caption="Encoder: GT image",
                        #     )
                        # )
                        
                        # predicted image
                        # resize the target image for visualization
                        viz_z_curr_decode = viz_tv_transforms_resize(z_curr_decode)
                        
                        
                        # predicted image
                        dec_pred_reward = self.conf_pred_output_act(pred_det_conf)
                        dec_pred_bboxes = pred_bboxes * img_data_h_w[0] if pred_bboxes is not None else None
                        # print(f"Pred bbox: {torch.max(dec_pred_bboxes, dim=0)}")
                        decoded_imgs.append(
                            wandb.Image(
                                data_or_path=viz_z_curr_decode[rand_img_idx]
                                .moveaxis(-3, -1)
                                .detach()
                                .cpu()
                                .numpy(),
                                caption="Encoder: Predicted image",
                                boxes={
                                    "predictions": {
                                        "box_data": [
                                            {
                                                "position": {
                                                    "minX": dec_pred_bboxes[rand_img_idx, 0].item(),
                                                    "maxX": dec_pred_bboxes[rand_img_idx, 2].item(),
                                                    "minY": dec_pred_bboxes[rand_img_idx, 1].item(),
                                                    "maxY": dec_pred_bboxes[rand_img_idx, 3].item(),
                                                }  \
                                                    if pred_bboxes is not None else \
                                                        {
                                                            "minX": img_data_h_w[0] * 0.3,
                                                            "maxX": img_data_h_w[0] * 0.6,
                                                            "minY": img_data_h_w[0] * 0.3,
                                                            "maxY": img_data_h_w[0] * 0.6,
                                                        },
                                                "domain": "pixel",
                                                "class_id": 1,
                                                "box_caption": f"{self.train_dataset.target_objects[sel_target_obj_idx[rand_img_idx]]}: {dec_pred_reward[rand_img_idx].item():.2f}", # TODO: Update this code.
                                                "scores": {"conf": dec_pred_reward[rand_img_idx].item() if self.rewards_predictor is not None
                                                           else -1}
                                            }
                                        ],
                                        # "class_labels": None, # TODO: Update this code.
                                    }
                                }
                            )
                        )
                        stacked_decoded_imgs.append(decoded_imgs)
                        
                        # pred_img_rgb = (
                        #     viz_z_curr_decode[rand_img_idx]
                        #         .moveaxis(-3, -1)
                        #         .detach()
                        #         .cpu()
                        #         .numpy()
                        #         * 255
                        # ).squeeze().astype(np.uint8)
                        
                        # # annotate bounding-box
                        # # RGB-to-BGR
                        # cv2_img = cv2.cvtColor(pred_img_rgb, cv2.COLOR_RGB2BGR)
                        # # cv2_img = pred_img_rgb[..., ::-1]
                        # pred_img_bboxes = pred_bboxes.detach().cpu().numpy()
                        # cv2.rectangle(
                        #     cv2_img,
                        #     (int(pred_img_bboxes[rand_img_idx][0]),
                        #      int(pred_img_bboxes[rand_img_idx][1])),
                        #     (int(pred_img_bboxes[rand_img_idx][2]),
                        #      int(pred_img_bboxes[rand_img_idx][3])),
                        #     (255, 0, 0),
                        #     2
                        # )
                        # # BGR-to-RGB
                        # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        # # cv2_img = cv2_img[..., ::-1]
                        # cv2_img = cv2_img.astype(float) / 255.0
                        
                        # decoded_imgs.append(
                        #     wandb.Image(
                        #         data_or_path=cv2_img[None],
                        #         caption="Encoder: Predicted image",
                        #     )
                        # )
                        # stacked_decoded_imgs.append(decoded_imgs)

                    if self.dynamics_predictor is not None:
                        if self.decoder is not None:
                            # random image index for visualizing the decoded image
                            rand_img_idx = torch.randint(
                                low=0,
                                high=resz_curr_img.shape[0],
                                size=(1,),
                            )
                            
                        # TODO: We can improve how we incorporate history.
                        # initialize the latent state history
                        z_hist = deque([z_curr] * (self.latent_state_history_length - 1))
                               
                        # loop over the observation sequence (timestep t1 to t_seq - 1)
                        for i in range(1, self.sequence_length):
                            # next image at t(i)
                            img_next = observation_sequence[i].to(self.device, non_blocking=True)
                            
                            # detection score
                            score_next = detection_confidence_scores[i].to(self.device, non_blocking=True)
                            
                            # bounding box
                            bbox_next = bboxes[i].to(self.device, non_blocking=True)
                            
                            # action
                            action_curr = actions[i - 1].to(self.device, non_blocking=True, dtype=img_curr.dtype)
                            
                            # next latent state as predicted by the target encoder
                            z_next_target = forward_target(img_next).to(img_next.dtype)
                            
                            # latent state with history
                            z_curr_with_history = torch.cat(
                                (z_curr, *z_hist),
                                dim=-2,
                            )
                            
                            # next latent state as predicted by the dynamics predictor
                            z_next_pred = self.dynamics_predictor(z_curr_with_history, action_curr[:, None])[..., :z_curr.shape[-2], :]
                            z_next_pred_mu, z_next_pred_logvar = z_next_pred[..., :self.dynamics_predictor.input_output_embed_dim], z_next_pred[..., self.dynamics_predictor.input_output_embed_dim:]

                            # sample from the distribution of the next latent state
                            z_next_pred_std = torch.exp(z_next_pred_logvar / 2.0)
                            z_next_pred = z_next_pred_mu + torch.randn_like(z_next_pred_std) * z_next_pred_std
                        
                            # compute the dynamics loss
                            dynamics_loss += (
                                loss_fn(z_next_pred, z_next_target)
                                + kl_divergence_weight * kl_divergence_loss_fn(
                                    z_next_pred_mu, z_next_pred_logvar
                                )
                            )
                        
                            if self.rewards_predictor is not None:
                                # next latent state from the next observation computed by the encoder
                                
                                if self.encoder_frozen:
                                    with torch.no_grad():
                                        z_next_enc = self.encoder(img_next).to(img_next.dtype)
                                else:
                                    z_next_enc = self.encoder(img_next).to(img_next.dtype)
                                
                                # # reshape, if necessary (for CLIP)
                                # if len(z_next_enc.shape) > 3:
                                #     z_next_enc = z_next_enc.reshape(
                                #         z_next_enc.shape[0],
                                #         z_next_enc.shape[-3] * z_next_enc.shape[-2],
                                #         z_next_enc.shape[-1],
                                # )
                                
                                # rewards from next observation computed by the encoder
                                pred_det_conf, pred_bboxes = self.rewards_predictor(z_next_enc, cond=cond_embeds)
  
                                # sample from the distribution of the rewards
                                pred_det_conf_mean, pred_det_conf_logvar = pred_det_conf[..., 0:1], pred_det_conf[..., 1:2]
                                pred_det_conf_std = torch.exp(pred_det_conf_logvar / 2.0)
                                pred_det_conf = pred_det_conf_mean + torch.randn_like(pred_det_conf_std) * pred_det_conf_std
                                
                                # compute the detection confidence loss
                                det_conf_loss = (
                                    conf_loss_fn(
                                        pred_det_conf.float(), score_next[:, None].float()
                                    )
                                    + kl_divergence_weight * kl_divergence_loss_fn(
                                        pred_det_conf_mean, pred_det_conf_logvar
                                    )
                                )
                                
                                # total detection loss
                                detection_loss += det_conf_loss
                                
                                if pred_bboxes is not None:
                                    # apply the sigmoid function to the predicted bounding boxes
                                    pred_bboxes = self.sigmoid(pred_bboxes)
                            
                                    # compute the loss for the predicted bounding boxes (assuming square images)
                                    bbox_loss += bbox_loss_fn(
                                        pred_bboxes.float(),
                                        bbox_next.float()
                                    )
                                    
                                    # # compute the loss for the predicted bounding boxes
                                    # # bbox_loss += bbox_loss_fn(pred_bboxes.float(), bbox_next.float())
                                    # bbox_loss += conf_loss_fn(pred_bboxes.float(), bbox_next.float())
                            
                                    # # apply the sigmoid function to the predicted bounding boxes
                                    # pred_bboxes = self.conf_pred_output_act(pred_bboxes)
                        
                                # rewards from predicted next observation
                                # decouple (detach) the dynamics predictor parameters from the rewards predictor parameters
                                pred_det_conf_align, pred_bboxes_align = self.rewards_predictor(z_next_pred.detach(), cond=cond_embeds)
            
                                # sample from the distribution of the rewards
                                pred_det_conf_align_mean, pred_det_conf_align_logvar = pred_det_conf_align[..., 0:1], pred_det_conf_align[..., 1:2]
                                pred_det_conf_align_std = torch.exp(pred_det_conf_align_logvar / 2.0)
                                pred_det_conf_align = pred_det_conf_align_mean + torch.randn_like(pred_det_conf_align_std) * pred_det_conf_align_std
                                
                                # compute the detection confidence loss
                                det_conf_loss_align = (
                                    conf_loss_fn(
                                        pred_det_conf_align.float(), score_next[:, None].float()
                                    )
                                    + kl_divergence_weight * kl_divergence_loss_fn(
                                        pred_det_conf_align_mean, pred_det_conf_align_logvar
                                    )
                                )
                                
                                # total detection loss
                                detection_loss += det_conf_loss_align
                                
                                if pred_bboxes_align is not None:
                                    # apply the sigmoid function to the predicted bounding boxes
                                    pred_bboxes_align = self.sigmoid(pred_bboxes_align)
                            
                                    # compute the loss for the predicted bounding boxes (assuming square images)
                                    bbox_loss += bbox_loss_fn(
                                        pred_bboxes_align.float(),
                                        bbox_next.float()
                                    )
                                    
                                    # # compute the loss for the predicted bounding boxes
                                    # # bbox_loss += bbox_loss_fn(pred_bboxes_align.float(), bbox_next.float())
                                    # bbox_loss += conf_loss_fn(pred_bboxes_align.float(), bbox_next.float())
                            
                                    # # apply the sigmoid function to the predicted bounding boxes
                                    # pred_bboxes_align = self.conf_pred_output_act(pred_bboxes_align)
                        
                            # decode the latent states
                            if self.decoder is not None:
                                # Remove the gradients of the inputs from the computation graph
                                # so we can train the decoder and the encoder independently.
                                z_next_pred_decode, _ = self.decoder(
                                    z_next_pred.detach().unsqueeze(1)
                                )

                                # reshape
                                z_next_pred_decode = z_next_pred_decode.reshape(
                                    img_next.shape[0], -1, *z_next_pred_decode.shape[-3:]
                                ).squeeze()

                                if tv_transforms_resize is None:
                                    # initialize torchvision transform Resize
                                    tv_transforms_resize = tv_transforms.Resize(
                                        z_next_pred_decode.shape[-2:]
                                    )

                                # resize the target image for the decoder loss
                                resz_next_img = tv_transforms_resize(img_next)

                                # decoder loss
                                decoder_loss += decoder_loss_fn(
                                    z_next_pred_decode, resz_next_img
                                )

                                # decoded and ground-truth states
                                decoded_imgs = []
                                
                                # resize the target image for visualization
                                viz_resz_next_img = viz_tv_transforms_resize(img_next)
                        
                                # ground-truth image
                                dec_bbox_next = bbox_next * img_data_h_w[0]
                                decoded_imgs.append(
                                    wandb.Image(
                                        data_or_path=viz_resz_next_img[rand_img_idx]
                                        .moveaxis(-3, -1)
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                        caption=f"Dynamics: GT image (t = {i})",
                                        boxes={
                                            "predictions": {
                                                "box_data": [
                                                    {
                                                        "position": {
                                                            "minX": dec_bbox_next[rand_img_idx, 0].item(),
                                                            "maxX": dec_bbox_next[rand_img_idx, 2].item(),
                                                            "minY": dec_bbox_next[rand_img_idx, 1].item(),
                                                            "maxY": dec_bbox_next[rand_img_idx, 3].item(),
                                                        },
                                                        "domain": "pixel",
                                                        "class_id": 1,
                                                        "box_caption": f"{self.train_dataset.target_objects[sel_target_obj_idx[rand_img_idx]]}: {score_next[rand_img_idx].item():.2f}", # TODO: Update this code.
                                                        "scores": {"conf": score_next[rand_img_idx].item()}
                                                    }
                                                ],
                                                # "class_labels": None, # TODO: Update this code.
                                            }
                                        }
                                    )
                                )
                                
                                # # ground-truth image
                                # gt_img_rgb = (
                                #     viz_resz_next_img[rand_img_idx]
                                #         .moveaxis(-3, -1)
                                #         .detach()
                                #         .cpu()
                                #         .numpy()
                                #         * 255
                                # ).squeeze().astype(np.uint8)
                        
                                # # annotate bounding-box
                                # # RGB-to-BGR
                                # cv2_img = cv2.cvtColor(gt_img_rgb, cv2.COLOR_RGB2BGR)
                                # # cv2_img = gt_img_rgb[..., ::-1]
                                # gt_img_bboxes = bbox_next.detach().cpu().numpy()
                                # cv2.rectangle(
                                #     cv2_img,
                                #     (int(gt_img_bboxes[rand_img_idx][0]),
                                #     int(gt_img_bboxes[rand_img_idx][1])),
                                #     (int(gt_img_bboxes[rand_img_idx][2]),
                                #     int(gt_img_bboxes[rand_img_idx][3])),
                                #     (255, 0, 0),
                                #     2
                                # )
                                # # BGR-to-RGB
                                # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                                # # cv2_img = cv2_img[..., ::-1]
                                # cv2_img = cv2_img.astype(float) / 255.0
                                
                                # decoded_imgs.append(
                                #     wandb.Image(
                                #         data_or_path=cv2_img[None],
                                #         caption=f"Dynamics: GT image (t = {i})",
                                #     )
                                # )
                                # predicted image
                                
                                # resize the pred image for visualization
                                viz_z_next_pred_decode = viz_tv_transforms_resize(z_next_pred_decode)
                        
                                # predicted image
                                dec_pred_reward_align = self.conf_pred_output_act(pred_det_conf_align)
                                dec_pred_bboxes_align = pred_bboxes_align * img_data_h_w[0] if pred_bboxes_align is not None else None
                                decoded_imgs.append(
                                    wandb.Image(
                                        data_or_path=viz_z_next_pred_decode[rand_img_idx]
                                        .moveaxis(-3, -1)
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                        caption=f"Dynamics: Predicted image (t = {i})",
                                        boxes={
                                            "predictions": {
                                                "box_data": [
                                                    {
                                                        "position": {
                                                            "minX": dec_pred_bboxes_align[rand_img_idx, 0].item(),
                                                            "maxX": dec_pred_bboxes_align[rand_img_idx, 2].item(),
                                                            "minY": dec_pred_bboxes_align[rand_img_idx, 1].item(),
                                                            "maxY": dec_pred_bboxes_align[rand_img_idx, 3].item(),
                                                        }  \
                                                            if pred_bboxes_align is not None else \
                                                                {
                                                                    "minX": img_data_h_w[0] * 0.3,
                                                                    "maxX": img_data_h_w[0] * 0.6,
                                                                    "minY": img_data_h_w[0] * 0.3,
                                                                    "maxY": img_data_h_w[0] * 0.6,
                                                                },
                                                        "domain": "pixel",
                                                        "class_id": 1,
                                                        "box_caption": f"{self.train_dataset.target_objects[sel_target_obj_idx[rand_img_idx]]}: {dec_pred_reward_align[rand_img_idx].item():.2f}", # TODO: Update this code.
                                                        "scores": {"conf": dec_pred_reward_align[rand_img_idx].item() if self.rewards_predictor is not None
                                                                else -1}
                                                    }
                                                ],
                                                # "class_labels": None, # TODO: Update this code.
                                            }
                                        }
                                    )
                                )
                                stacked_decoded_imgs.append(decoded_imgs)
                            
                            
                                # pred_img_rgb = (
                                #     viz_z_next_pred_decode[rand_img_idx]
                                #         .moveaxis(-3, -1)
                                #         .detach()
                                #         .cpu()
                                #         .numpy()
                                #         * 255
                                # ).squeeze().astype(np.uint8)
                                
                                # # annotate bounding-box
                                # # RGB-to-BGR
                                # cv2_img = cv2.cvtColor(pred_img_rgb, cv2.COLOR_RGB2BGR)
                                # # cv2_img = pred_img_rgb[..., ::-1]
                                # pred_img_bboxes = pred_bboxes_align.detach().cpu().numpy()
                                # cv2.rectangle(
                                #     cv2_img,
                                #     (int(pred_img_bboxes[rand_img_idx][0]),
                                #     int(pred_img_bboxes[rand_img_idx][1])),
                                #     (int(pred_img_bboxes[rand_img_idx][2]),
                                #     int(pred_img_bboxes[rand_img_idx][3])),
                                #     (255, 0, 0),
                                #     2
                                # )
                                # # BGR-to-RGB
                                # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                                # # cv2_img = cv2_img[..., ::-1]
                                # cv2_img = cv2_img.astype(float) / 255.0
                                
                                # decoded_imgs.append(
                                #     wandb.Image(
                                #         data_or_path=cv2_img[None],
                                #         caption=f"Dynamics: Predicted image (t = {i})",
                                #     )
                                # )
                                # stacked_decoded_imgs.append(decoded_imgs)
                            
                            # update the latent state history
                            if self.latent_state_history_length > 1:    
                                z_hist.pop() # remove from right
                                z_hist.appendleft(z_curr) # append to left

                            # advance the latent state computed by the dynamics predictor
                            z_curr = z_next_pred

                    logdict = {
                        "dynamics_loss": dynamics_loss,
                        "detection_loss": detection_loss,
                        "decoder_loss": decoder_loss,
                        "bbox_loss": bbox_loss,
                        "epoch": epoch,
                    }

                    # decoded images
                    if log_images: 
                        decoded_imgs_dict = {
                            f"image_{w_idx}": w_img
                            for w_idx, w_img in enumerate(stacked_decoded_imgs)
                        }
                        logdict.update(decoded_imgs_dict)

                    wandb.log(logdict)

                    # total loss
                    loss = (
                        dynamics_loss 
                        + rewards_loss_weight * detection_loss 
                        + bbox_loss_weight * bbox_loss 
                        + decoder_loss
                    )

                    # update params
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.scheduler.step()
                    self.wd_scheduler.step()
                    
                    assert not torch.isnan(loss), "loss is nan"

                    # Step 3. momentum update of target encoder
                    with torch.no_grad():
                        m = next(self.momentum_scheduler)
                        for param_q, param_k in zip(
                            self.encoder.parameters(), self.target_encoder.parameters()
                        ):
                            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                # train the model
                train_step()

            # save the checkpoint
            self.save_checkpoint(epoch + 1)

    def ablate_dynamics(self):
        # examine only the dynamics predictor
        self.ablation = "dynamics"
        del self.rewards_predictor
        torch.cuda.empty_cache()
        gc.collect()
        self.rewards_predictor = None
    
    def ablate_rewards(self):
        # examine only the rewards predictor
        self.ablation = "rewards"
        del self.dynamics_predictor
        torch.cuda.empty_cache()
        gc.collect()
        self.dynamics_predictor = None
     
    def simple_gradient_planner(self, z, steps=10, action_lr=1e-3):
        a = torch.zeros(6).to(self.device)
        a = a.unsqueeze(0).unsqueeze(0)
    
        for _ in range(steps):
            z_next = self.dynamics_predictor(z, a)
            raw_reward = self.rewards_predictor(z_next)
            reward = self.conf_pred_output_act(raw_reward)
            grad = torch.zeros_like(a)

            eps = np.ones(6)
            eps[0:3] *= 2e-3
            eps[3:] *= 0.1
            
            for i in range(6):
                a_perturb = a.clone()
                a_perturb[0][0][i] = eps[i]
                z_next_perturb = self.dynamics_predictor(z, a_perturb)
                new_reward = self.conf_pred_output_act(self.rewards_predictor(z_next_perturb))
                grad[0][0][i] = (new_reward - reward) / eps[i]
            
            a += action_lr * grad
        
        predicted_znext = self.dynamics_predictor(z, a)
        predicted_rnext = self.conf_pred_output_act(self.rewards_predictor(predicted_znext))
        return a, predicted_znext, predicted_rnext
    
    def mpc_gradient_planner(self, z, horizon=5, steps=10, action_lr=1e-3):
        # Initialize a sequence of actions (horizon x 6) with zeros
        a_seq = torch.zeros(horizon, 6, device=self.device)
        a_seq = a_seq.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, 6)

        for _ in range(steps):  # Gradient ascent steps
            cumulative_reward = 0
            grad_seq = torch.zeros_like(a_seq)
            
            z_curr = z.clone()
            
            for t in range(horizon):
                # Predict the next latent state
                a = a_seq[:, :, t:t+1, :].squeeze(0)
                z_next = self.dynamics_predictor(z_curr, a)
                raw_reward = self.rewards_predictor(z_next)
                reward = self.conf_pred_output_act(raw_reward)
                cumulative_reward += reward

                # Compute gradient for action at step t
                grad = torch.zeros_like(a)

                eps = np.ones(6)
                eps[0:3] *= 1e-3
                eps[3:] *= 0.03
                
                for i in range(6):
                    a_perturb = a_seq.clone()
                    a_perturb[:, :, t, i] += eps[i]
                    
                    z_next_perturb = self.dynamics_predictor(z_curr, a_perturb[:, :, t:t+1, :].squeeze(0))
                    new_reward = self.conf_pred_output_act(self.rewards_predictor(z_next_perturb))
                    
                    grad[:, :, i] = (new_reward - reward) / eps[i]
                
                grad_seq[:, t:t+1, :] = grad
                z_curr = z_next  # Update current state for the next time step

            # Gradient ascent step for all actions in the sequence
            a_seq += action_lr * grad_seq.squeeze(0)

        # Execute only the first action and return its prediction
        a_final = a_seq[:, :, 0, :]  # Take the first action
        predicted_znext = self.dynamics_predictor(z, a_final)
        predicted_rnext = self.conf_pred_output_act(self.rewards_predictor(predicted_znext))
        
        return a_final, predicted_znext, predicted_rnext
    
    def joint_gradient_planner(
        self,
        z,
        horizon_len=4, #5
        steps=10, #10
        action_lr=1e-3,#1e-3,
        target_obj_embed=None,
        action_initialization: Literal["zeros", "ones", "random"] = "random",
        init_action=None,
        enable_std_dev_weighted_rewards: bool = False,
        enable_scheduler: bool = False,
        verbose_print: bool = False,
        return_all_sequence: bool = False,
    ):
        """
        Returns:
            action: optimized action
            predicted_znext: predicted next latent state
            pred_reward: predicted reward
            pred_bboxes: predicted bounding boxes
            pred_reward_std: predicted reward standard deviation
        """
        
        # action initialization
        if init_action is not None:
            action = torch.tensor(init_action, dtype=torch.float32)  # (horizon_len, 6)
            
            if action.ndim != 2 or action.shape[1] != 6:
                raise ValueError("Provided action list must have shape (horizon_len, 6)")
            
            horizon_len = action.shape[0]
            action = action.unsqueeze(1)  # (horizon_len, 1, 6)
            # action = init_action.detach().clone()
        else:
            if action_initialization.lower() == "zeros":
                action = torch.zeros(horizon_len, 6).unsqueeze(1)
            elif action_initialization.lower() == "ones":
                action = 5e-3 * torch.ones(horizon_len, 6).unsqueeze(1)
            elif action_initialization.lower() == "random":
                action = 1e-2 * torch.randn(
                    horizon_len, 1, 6,
                    dtype=torch.float32,
                    device="cuda",
                )
                action[:3] *= 2 # TODO change
        action = torch.nn.Parameter(action.to(self.device))
        
        # optimizer
        optimizer = torch.optim.Adam([action], lr=action_lr)

        if enable_scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        
        # initial latent state
        z_init = z.detach().clone()
        
        for iter_idx in range(steps):
            
            # initialize the loss
            loss = 0.0
            
            # reinitialize the latent state
            z = z_init.clone()
                
            # TODO: We can improve how we incorporate history.
            # initialize the latent state history
            z_hist = deque([z] * (self.latent_state_history_length - 1))
            
            for hor_idx in range(horizon_len):
                
                # latent state with history
                z_curr_with_history = torch.cat(
                    (z, *z_hist),
                    dim=-2,
                )
                
                # predict the next latent state
                z = self.dynamics_predictor(
                    z_curr_with_history,
                    action=action[hor_idx].unsqueeze(dim=0),
                )[..., :z.shape[-2], :]
                
                z_mu, z_logvar = z[..., :self.dynamics_predictor.input_output_embed_dim], z[..., self.dynamics_predictor.input_output_embed_dim:]

                # sample from the distribution of the next latent state
                z_std = torch.exp(z_logvar / 2.0)
                z = z_mu + torch.randn_like(z_std) * z_std
                        
                # rewards predictor
                pred_reward, pred_bboxes = self.rewards_predictor(z, cond=target_obj_embed)

                # sample from the distribution of the rewards
                pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
                pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
                pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                
                pred_reward = self.conf_pred_output_act(pred_reward)
                
                if enable_std_dev_weighted_rewards:
                    # optimize over the standard deviation-weighted rewards
                    # weight the rewards
                    weighted_pred_reward = pred_reward / pred_reward_std
                else:
                    # use a weught of one
                    weighted_pred_reward = pred_reward
                   
                # loss function
                loss += torch.abs(1.0 - weighted_pred_reward)
                
                # update the latent state history
                if self.latent_state_history_length > 1:
                    z_hist.pop() # remove from right
                    z_hist.appendleft(z) # append to left
                     
                
            # step the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if enable_scheduler:
                scheduler.step()
        
        # TODO refactor
        if not return_all_sequence:
            # optimized action
            action = action.detach()[0].unsqueeze(dim=0)
            
            # predicted next latent state and rewards
            # initialize the latent state history
            z_hist = deque([z_init] * (self.latent_state_history_length - 1))
                
            z_curr_with_history = torch.cat(
                (z_init, *z_hist),
                dim=-2,
            )
            
            z_next = self.dynamics_predictor(z_curr_with_history, action)[..., :z.shape[-2], :]
            
            z_next_mu, z_next_logvar = z_next[..., :self.dynamics_predictor.input_output_embed_dim], z_next[..., self.dynamics_predictor.input_output_embed_dim:]
            
            # sample from the distribution of the next latent state
            z_next_std = torch.exp(z_next_logvar / 2.0)
            predicted_znext = z_next_mu + torch.randn_like(z_next_std) * z_next_std
                    
            # rewards predictor
            pred_reward, pred_bboxes = self.rewards_predictor(predicted_znext, cond=target_obj_embed)
            
            # sample from the distribution of the rewards
            pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
            pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
            pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                    
            pred_reward = self.conf_pred_output_act(pred_reward)
            
            if init_action is not None and verbose_print:
                print(f"Init-and-optimzed action difference norm: {torch.linalg.norm(init_action[0].squeeze() - action.squeeze())}")
                    
            if pred_bboxes is not None:
                # apply the sigmoid function to the predicted bounding boxes
                pred_bboxes = self.sigmoid(pred_bboxes)
                            
                # apply the sigmoid function to the predicted bounding boxes
                pred_bboxes = pred_bboxes.squeeze()
                                
            return action.squeeze(), predicted_znext, pred_reward, pred_bboxes, pred_reward_std
        else:
            # optimized actions
            action_list = action.detach()
            
            z = z_init.clone()
            predicted_znexts = []
            pred_rewards = []
            pred_bboxes = [] 
            pred_reward_stds = []
            
            for act in action_list:
                # initialize the latent state history
                z_hist = deque([z] * (self.latent_state_history_length - 1))
                    
                z_curr_with_history = torch.cat(
                    (z, *z_hist),
                    dim=-2,
                )
                
                z_next = self.dynamics_predictor(z_curr_with_history, act.unsqueeze(0))[..., :z.shape[-2], :]
                
                z_next_mu, z_next_logvar = z_next[..., :self.dynamics_predictor.input_output_embed_dim], z_next[..., self.dynamics_predictor.input_output_embed_dim:]
                
                # sample from the distribution of the next latent state
                z_next_std = torch.exp(z_next_logvar / 2.0)
                predicted_znext = z_next_mu + torch.randn_like(z_next_std) * z_next_std
                
                predicted_znexts.append(predicted_znext)
                
                # rewards predictor
                pred_reward, pred_bboxes = self.rewards_predictor(predicted_znext, cond=target_obj_embed)
                
                # sample from the distribution of the rewards
                pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
                pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
                pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                        
                pred_reward = self.conf_pred_output_act(pred_reward)
                
                pred_rewards.append(pred_reward.item())
                pred_reward_stds.append(pred_reward_std.item())
                
            if pred_bboxes is not None:
                # apply the sigmoid function to the predicted bounding boxes
                pred_bboxes = self.sigmoid(pred_bboxes)
                            
                # apply the sigmoid function to the predicted bounding boxes
                pred_bboxes = pred_bboxes.squeeze()
                pred_bboxes.append(pred_bboxes)
                                
            return action_list, predicted_znexts, pred_rewards, pred_bboxes, pred_reward_stds