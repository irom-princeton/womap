import torch
import torch.nn.functional as F
import numpy as np
from src.utils.distributed import AllReduce


def cross_entropy_loss_fn(input, target, all_reduce=True):
    loss = F.binary_cross_entropy_with_logits(
        input, target
    )
    if all_reduce:
        loss = AllReduce.apply(loss)
    return loss

def smooth_l1_loss_fn(z, h, beta=1.0, all_reduce=True):
    loss = F.smooth_l1_loss(z, h, beta=beta)
    if all_reduce:
        loss = AllReduce.apply(loss)
    return loss

def mse_loss_fn(z, h, all_reduce=True):
    loss = F.mse_loss(z, h)
    if all_reduce:
        loss = AllReduce.apply(loss)
    return loss

def kl_divergence_loss_fn(mu, logvar, all_reduce=True):
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    loss = loss.mean()
    if all_reduce:
        loss = AllReduce.apply(loss)
    return loss

def ciou_loss_fn(z, h, reduce="mean", all_reduce=True):
    # error-checking
    # assert torch.all(
    #     torch.logical_and(
    #         z[..., 0] <= z[..., 2],
    #         z[..., 1] <= z[..., 3]
    #     )
    # )
    
    # epsilon to avoid division-by-zero error
    eps = 1e-12
    
    # intersection of the bounding-boxes
    bbox_intersect = torch.stack(
        (
            torch.maximum(z[..., 0], h[..., 0]),
            torch.maximum(z[..., 1], h[..., 1]),
            torch.minimum(z[..., 2], h[..., 2]),
            torch.minimum(z[..., 3], h[..., 3]),
        ),
        dim=-1,
    )
    
    # no intersection?
    no_intersect_mask = torch.logical_or(
        (bbox_intersect[..., 2] < bbox_intersect[..., 0]),
        (bbox_intersect[..., 3] < bbox_intersect[..., 1]),
    )
    
    # IoU loss
    iou_loss = torch.zeros_like(z[..., 0]).to(z.device)
    
    # area of the intersection and union
    intersect_area = (
        (bbox_intersect[..., 2] - bbox_intersect[..., 0])
        * (bbox_intersect[..., 3] - bbox_intersect[..., 1])
    )
    
    union_area = (
        (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
        + (h[..., 2] - h[..., 0]) * (h[..., 3] - h[..., 1])
    )
    
    # IoU loss
    iou_loss = intersect_area / (union_area + eps)
    
    # no intersection
    iou_loss[no_intersect_mask] *= 0.0
    
    # smallest bounding-box (i.e., convex shape) for the union
    convex_bbox = torch.stack(
        (
            torch.minimum(z[..., 0], h[..., 0]),
            torch.minimum(z[..., 1], h[..., 1]),
            torch.maximum(z[..., 2], h[..., 2]),
            torch.maximum(z[..., 3], h[..., 3]),
        ),
        dim=-1
    )
    
    # centroid distance
    centroid_z = torch.stack(
        (
            (z[..., 0] + z[..., 2]) / 2,
            (z[..., 1] + z[..., 3]) / 2,
        ),
        dim=-1,
    )
    
    centroid_h = torch.stack(
        (
            (h[..., 0] + h[..., 2]) / 2,
            (h[..., 1] + h[..., 3]) / 2,
        ),
        dim=-1,
    )
    
    # centroid distance
    centroid_dist = torch.sum((centroid_z - centroid_h) ** 2.0, dim=-1)
    
    # diagonal length of the convex shape (bounding-box)
    convex_bbox_diag_len = (
        (convex_bbox[..., 2] - convex_bbox[..., 0]) ** 2.0
        + (convex_bbox[..., 3] - convex_bbox[..., 1]) ** 2.0 
    )
    
    # width and height of the ground-truth and predicted bounding-boxes
    gt_bbox_width = h[..., 2] - h[..., 0]
    gt_bbox_height = h[..., 3] - h[..., 1]
    pred_bbox_width = z[..., 2] - z[..., 0]
    pred_bbox_height = z[..., 3] - z[..., 1]
    
    # aspect ratio component
    aspect_ratio_comp = 4 / (np.pi ** 2.0) * (
        torch.atan2(gt_bbox_width, gt_bbox_height)
        - torch.atan2(pred_bbox_width, pred_bbox_height)
    ) ** 2.0
    
    # aspect ratio component weight
    aspect_ratio_comp_weight = (
        aspect_ratio_comp 
        / ((1 - iou_loss) + aspect_ratio_comp)
    )
      
    # complete IoU loss
    loss = (
        1 - iou_loss 
        + (centroid_dist / convex_bbox_diag_len) 
        + aspect_ratio_comp_weight * aspect_ratio_comp 
    )
    
    if reduce.lower() == "mean":
        loss = loss.mean()
    elif reduce.lower() == "sum":
        loss = loss.sum()
    
    if all_reduce:
        loss = AllReduce.apply(loss)
    return loss
