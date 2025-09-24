import torch
import math
from torchvision import ops

def loss(predictions, targets, lambda_coord=5,lambda_noobj=.5, S=7, B=2, C=20):
    batch_size = predictions.shape[0]

    pred_boxes = predictions[...,C:].reshape(batch_size, S, S, B, 5)
    target_boxes = targets[...,C:].reshape(batch_size, S, S, 1, 5)


    # extract 5 regression variables
    pred_confidence, target_confidence = pred_boxes[...,0], target_boxes[...,0]
    pred_xywh = pred_boxes[..., 1:5]
    target_xywh = target_boxes[..., 1:5]

    # extract class probabilities
    pred_classes, target_classes = predictions[...,:C], targets[..., :C]

    # format boxes for iou calculation
    pred_boxes_iou = convert_box_iou_format(pred_xywh)
    target_boxes_iou = convert_box_iou_format(target_xywh[..., 0, :])

    ious = torch.zeros(batch_size, S, S, B, device=predictions.device)
    for i in range(B):
        ious[..., i] = ops.box_iou(
            pred_boxes_iou[..., i, :].view(-1, 4),
            target_boxes_iou.view(-1, 4)
        ).diag().view(batch_size, S, S)
    
    iou_maxes, best_box_indices = torch.max(ious, dim=-1, keepdim=True)

    exists_box = targets[..., C:C+1] > 0 

    obj_mask = exists_box * (torch.zeros_like(ious).scatter_(-1, best_box_indices, 1))
    noobj_mask = (1 - obj_mask) * exists_box.float() 
    noobj_mask += (1 - exists_box.any(-1, keepdim=True).float()) 

    pred_confidence, target_confidence = pred_boxes[..., 0], iou_maxes # Target is the actual IoU
    pred_xy, target_xy = pred_boxes[..., 1:3], target_boxes[..., 1:3]

    pred_wh = torch.sign(pred_boxes[..., 3:5]) * torch.sqrt(torch.abs(pred_boxes[..., 3:5]) + 1e-6)
    target_wh = torch.sqrt(target_boxes[..., 3:5])

    # Term 1: Localization Loss (xy)
    xy_loss = lambda_coord * torch.sum(obj_mask[..., None] * torch.square(target_xy - pred_xy))

    # Term 2: Localization Loss (wh)
    wh_loss = lambda_coord * torch.sum(obj_mask[..., None] * torch.square(target_wh - pred_wh))

    # Term 3: Confidence Loss (Object)
    # The target is the actual IoU value for the best box.
    conf_obj_loss = torch.sum(obj_mask * torch.square(target_confidence - pred_confidence))

    # Term 4: Confidence Loss (No Object)
    # Target confidence here is 0.
    conf_noobj_loss = lambda_noobj * torch.sum(noobj_mask * torch.square(0 - pred_confidence))

    # Term 5: Classification Loss
    cell_obj_mask = exists_box.bool().any(-1, keepdim=True)
    class_loss = torch.sum(cell_obj_mask * torch.square(target_classes - pred_classes))

    # --- Step 5: Final Loss ---
    # The total loss is the sum of the individual components.
    total_loss = xy_loss + wh_loss + conf_obj_loss + conf_noobj_loss + class_loss
    
    return total_loss

def convert_box_iou_format(boxes):
    x, y = boxes[..., 0], boxes[..., 1]
    w, h = boxes[..., 2], boxes[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
    