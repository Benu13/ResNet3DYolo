# %%
from typing import Optional, Tuple, Union, List, Sequence
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
import os

DeviceType = Union[str, torch.device]
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model



def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w, d = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift z

        sy, sx, sz = torch.meshgrid(sy, sx, sz, indexing='ij')
        anchor_points.append(torch.stack((sx, sy, sz), -1).view(-1, 3))
        stride_tensor.append(torch.full((h * w * d, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 3, dim)
    x1y1z1 = anchor_points - lt
    x2y2z2 = anchor_points + rb
    return torch.cat((x1y1z1, x2y2z2), dim)  # xyzxyz bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1z1, x2y2z2 = torch.split(bbox, 3, -1)
    return torch.cat((anchor_points - x1y1z1, x2y2z2 - anchor_points), -1).clamp(0, reg_max - 0.1)  # dist (lt, rb)

def bbox_iou(box1, box2, eps = 1e-7, iou_mode:bool=False):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2= box1.chunk(6, -1)
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2.chunk(6, -1)

    # Union
    w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
    w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0) * \
            (torch.min(b1_z2, b2_z2) - torch.max(b1_z1, b2_z1)).clamp(0)

    # Union Area
    union = (w1 * h1 * d1 ) + (w2 * h2 * d2) - inter + eps

    # IoU
    iou = inter / union
    if iou_mode:
        return iou.clamp(min=0., max=1.0)
    
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    cd = torch.max(b1_z2, b2_z2) - torch.min(b1_z1, b2_z1)  # convex depth

    c2 = cw ** 2 + ch ** 2 + cd**2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2

    a = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2+eps)) - torch.atan(w1 / (h1+eps)), 2)
    d = (4 / (math.pi**2)) * torch.pow(torch.atan(d2 / (h2+eps)) - torch.atan(d1 / (h1+eps)), 2)
    v = (a+d)/2
    
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 6)
            anc_points (Tensor): shape(num_total_anchors, 3)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 6)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """

        self.bs = pd_scores.size(0) #batch size
        self.n_max_boxes = gt_bboxes.size(1) # max boxes that can be present on image == num classes 
        
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        #print(target_scores)
        #print(align_metric)
        #raise
        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        #print(target_scores)
        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        # get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (b, max_num_obj, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        # compute alignment for gt and boxes
        gt_labels = gt_labels.to(torch.long)  # b, max_num_obj, 1
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls

        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1)).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        """
        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        # assigned topk should be unique, this is for dealing with empty labels
        # since empty labels will generate index `0` through `F.one_hot`
        # NOTE: but what if the topk_idxs include `0`?
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """

        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 6)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
    
################################################################## LOSS #################################################################

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_overlaps(pred: torch.Tensor,
                  target: torch.Tensor,
                  eps: float = 1e-7) -> torch.Tensor:
    r"""
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """

    bbox1_x1, bbox1_y1, bbox1_z1 = pred[..., 0], pred[..., 1], pred[..., 2]
    bbox1_x2, bbox1_y2, bbox1_z2 = pred[..., 3], pred[..., 4], pred[..., 5]

    bbox2_x1, bbox2_y1, bbox2_z1 = target[..., 0], target[..., 1], target[..., 2]
    bbox2_x2, bbox2_y2, bbox2_z2 = target[..., 3], target[..., 4], target[..., 5]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)* \
              (torch.min(bbox1_z2, bbox2_z2) -
               torch.max(bbox1_z1, bbox2_z1)).clamp(0)

    # Union
    w1, h1, d1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1, bbox1_z2 - bbox1_z1
    w2, h2, d2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1, bbox2_z2 - bbox2_z1
    union = (w1 * h1 * d1) + (w2 * h2 * d2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1z1 = torch.min(pred[..., :3], target[..., :3])
    enclose_x2y2z2 = torch.max(pred[..., 3:], target[..., 3:])
    enclose_whd = (enclose_x2y2z2 - enclose_x1y1z1).clamp(min=0)

    enclose_w = enclose_whd[..., 0]  # cw
    enclose_h = enclose_whd[..., 1]  # ch
    enclose_d = enclose_whd[..., 2]  # cd
    # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )
    # calculate enclose area (c^2)
    enclose_area = enclose_w**2 + enclose_h**2 + enclose_d**2 + eps

    # calculate ρ^2(b_pred,b_gt):
    # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
    # center point, because bbox format is xyxy -> left-top xy and
    # right-bottom xy, so need to / 4 to get center point.
    rho2_left_item = ((bbox2_x1 + bbox2_x2) - 
                        (bbox1_x1 + bbox1_x2))**2 / 4
    rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                        (bbox1_y1 + bbox1_y2))**2 / 4
    rho2_front_item = ((bbox2_z1 + bbox2_z2) -
                        (bbox1_z1 + bbox1_z2))**2 / 4
    
    rho2 = rho2_left_item + rho2_right_item + rho2_front_item # rho^2 (ρ^2)

    # Width and height ratio (v)
    wh_ratio = (4 / (math.pi**2)) * torch.pow(
        torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    wd_ratio = (4 / (math.pi**2)) * torch.pow(
        torch.atan(d2 / h2) - torch.atan(d1 / h1), 2)
    
    ratio = (wh_ratio + wd_ratio )/2

    with torch.no_grad():
        alpha = ratio / (ratio - ious + (1 + eps))

    # CIoU
    ious = ious - ((rho2 / enclose_area) + (alpha * ratio))
    return ious.clamp(min=-1.0, max=1.0)

class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    
# ASSIGNER
def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 6)
        gt_bboxes (Tensor): shape(b, n_boxes, 6)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 6).chunk(2, 2)  # left-top-front, right-bottom-back
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w*d)
        overlaps (Tensor): shape(b, n_max_boxes, h*w*d)
    Return:
        target_gt_idx (Tensor): shape(b, h*w*d)
        fg_mask (Tensor): shape(b, h*w*d)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w*d)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos

class ComputeLoss:
    # Compute losses
    def __init__(self, m, h, use_dfl=True):
        device = h['device']

        # Define criteria
        weight = torch.tensor(h['weight'], dtype=torch.float).to(device) if 'weight' in list(h.keys()) else None
        BCEcls = nn.BCEWithLogitsLoss(weight=weight, pos_weight=None, reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        #self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.stride = m.stride # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.bw = h['bbox_weight'] if 'bbox_weight' in list(h.keys()) else 7.5
        self.cw = h['class_weight'] if 'class_weight' in list(h.keys()) else 0.5
        self.dw = h['dfl_weight'] if 'dfl_weight' in list(h.keys()) else 1.5

        self.assigner = TaskAlignedAssigner(topk=5,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0)
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def bbox_decode(self, anchor_points, pred_dist):
        # decode predicted distances to bboxes
        # 1. split distances into respective anchors and reg_maxes
        # 2. limit distance to positive values with softmax
        # 3. multiply by reg_maxes
        # return box in [x_min,y_min,z_min,x_max,y_max,z_max]
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels [b, h*w*c for h,w,c in feature levels, reg_max*6]
            # limit predicted distances to positive values and multiply each reg_max level by reg value
            pred_dist = pred_dist.view(b, a, 6, self.reg_max).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            #print(pred_dist)
        return dist2bbox(pred_dist, anchor_points)

    def __call__(self, p, gt_bboxes, gt_labels, img=None, epoch=0):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = p
        
        # flatten and conncat all level features and split back split preds [b, levels*box_pred+cls_pred, h*w*d] -> [b, levels*box_pred, h, w, d], [b, levels*cls_pred, h*w*d]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 6, self.nc), 1) 

        #s_scores = pred_scores.clone().detach()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #[b, h*w*d summed for all feature levels, reg_max*6]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #[b, h*w*d summed for all feature levels, nc]

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5) #generate anchors
        
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) # mask for present bboxes in given gt 1 if bbox is present, 0 otherwise

        # pboxes
        #print(pred_distri)
        
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyzxyz, (b, h*w*d, 6)
        
        #print((pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype))
        #raise
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)
        
        scores, labels = pred_scores.sigmoid().max(-1, keepdim=True)
        masked_a_preds = [torch.cat([
            torch.masked_select(score, fg_mask_i.unsqueeze(-1)).view(-1,1),
            torch.masked_select(label, fg_mask_i.unsqueeze(-1)).view(-1,1),
            torch.masked_select((pred_bbox.clone().detach() * stride_tensor), fg_mask_i.unsqueeze(-1).repeat([1, 6])).view(-1,6)], dim=-1) 
            for score, label, pred_bbox, fg_mask_i in zip(scores, labels, pred_bboxes, fg_mask)]

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        
        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        loss[0] *= self.bw  # box gain
        loss[1] *= self.cw # cls gain
        loss[2] *= self.dw  # dfl gain

        return loss.sum() * batch_size, loss.detach(), masked_a_preds  # loss(box, cls, dfl)
    
class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 6])  # (b, h*w*d, 6)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 6)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 6)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos)
        loss_iou = 1.0 - iou
        
        #print(bbox_weight)
        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum
    
        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 6])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 6, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 6)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class Nms3d(nn.Module):
    def __init__(self, iou_treshold=0.3, score_treshold = 0., del_dup_class=False) -> None:
        super().__init__()
        self.score_tr = score_treshold
        self.iou_tr = iou_treshold
        self.del_dup_class = del_dup_class

    def forward(self, results):
        # boxes in format [score, label, x, y, z, x, y, z]
        result = results.clone().detach()
        result = result[result[:,0].argsort(dim=0, descending=True)] # sort by score 
        result = result[result[:,0] >= self.score_tr] # score tr
        filtered = []
        while result.nelement() != 0:
            filtered.append(result[[0]])
            ious = bbox_iou(result[0, 2:], result[:, 2:], iou_mode=True)
            if self.del_dup_class:
                result = result[torch.logical_and(ious.squeeze()<=self.iou_tr, result[:,1]!=result[0, 1])]
            else:
                result = result[ious.squeeze()<=self.iou_tr]

        return torch.cat(filtered, dim=0)

class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 6, self.c1, a).transpose(2, 1).softmax(1)).view(b, 6, a)
    
class PredictionHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels) -> None:
        super().__init__()
        
        self.f_modules = nn.Sequential(
                                        nn.Conv3d(
                                        in_channels,
                                        mid_channels,
                                        kernel_size=(3,3,3),
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        ),
                                        nn.BatchNorm3d(mid_channels),
                                        nn.SiLU(inplace=True),
                                        nn.Conv3d(
                                        mid_channels,
                                        mid_channels,
                                        kernel_size=(3,3,3),
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False), 
                                        nn.BatchNorm3d(mid_channels),
                                        nn.SiLU(inplace=True),
                                        nn.Conv3d(
                                        mid_channels,
                                        out_channels,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False) 
                                        )

    def forward(self, x):
        x = self.f_modules(x)
        return x


class Detect3d(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=5, ch=(), strides = (), reg_max = 16, inplace=True, return_logits=False):  # detection layer
        super().__init__()
        self.return_logits = return_logits
        self.joint_training = False
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 6  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = strides#torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 6, self.reg_max * 6, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList([PredictionHead(x, c2, 6 * self.reg_max) for x in ch])
        self.cv3 = nn.ModuleList([PredictionHead(x, c3, nc) for x in ch])

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHWD
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]).permute(0,1,3,4,2), self.cv3[i](x[i]).permute(0,1,3,4,2)), 1)
        if self.training and not self.joint_training:
            return x
        #####
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 6, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), dim=1) * self.strides
        scores, labels = cls.sigmoid().max(1, keepdim=True)
        
        if self.return_logits:
            return scores, labels, dbox, cls
        return scores, labels, dbox
        #y = torch.cat((dbox, cls.sigmoid()), 1)
        #return y if self.export else (y, x)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (400 / stride)**2)
            