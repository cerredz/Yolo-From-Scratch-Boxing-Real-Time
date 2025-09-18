import torch
import math
from torchvision import ops

def loss(predictions, targets, lambda_coord=5,lambda_noobj=.5, S=7, B=2, C=20):
    batch_size = predictions.shape[0]
    print(predictions.shape)
    pred_boxes = predictions[...,C:].reshape(batch_size, S, S, B, 5)
    target_boxes = targets[...,C:].reshape(batch_size, S, S, B, 5)

    # extract 5 regression variables
    pred_confidence, target_confidence = pred_boxes[...,0], target_boxes[...,0]
    pred_xy, target_xy = pred_boxes[...,1:3], target_boxes[...,1:3]
    pred_wh, target_wh = pred_boxes[...,3:5], target_boxes[...,3:5]

    # extract class probabilities
    pred_classes, target_classes = predictions[...,:C], targets[..., :C]

    iou_val = iou(pred_boxes, target_boxes)

    obj_mask = target_confidence == iou_val
    noobj_mask = ~obj_mask
    cell_obj_mask = torch.any(obj_mask, dim=-1)

    # calculate term 1
    xy_loss = lambda_coord * torch.sum(obj_mask[..., None] * (target_xy - pred_xy) ** 2)

    # calculate term 2
    wh_diff = torch.sqrt(target_wh) - torch.sqrt(torch.abs(pred_wh))
    wh_loss = lambda_coord * torch.sum(obj_mask[..., None] * wh_diff ** 2)

    # calculate term 3
    conf_obj_loss = torch.sum(obj_mask * (target_confidence - pred_confidence) ** 2)

    # calculate term 4
    conf_noobj_loss = lambda_noobj * torch.sum(noobj_mask * (target_confidence - pred_confidence) ** 2)

    # calculate term 5
    class_loss = torch.sum(cell_obj_mask[..., None] * (target_classes - pred_classes) ** 2)

    loss_val = (loss - iou_val) ** 2

    return loss_val

def iou(pred_boxes, target_boxes):
    
    iou_tensor = ops.box_iou(target_boxes, pred_boxes)
    iou_value = torch.max(iou_tensor)
    return iou_value