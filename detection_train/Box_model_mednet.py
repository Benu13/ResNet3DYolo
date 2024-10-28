from typing import Optional, Tuple, Union, List, Sequence
from MedicalNet.MedicalNet import Struct, MedNet
import detection_train.yolov9_head_func_3d as y9
import torch
import torch.nn as nn

class Nms3d(nn.Module):
    def __init__(self, iou_treshold=0.3, score_treshold = 0., del_dup_class=False, normalize=False, series_dim = None) -> None:
        super().__init__()
        self.score_tr = score_treshold
        self.iou_tr = iou_treshold
        self.del_dup_class = del_dup_class
        self.normalize = normalize
        self.series_dim = series_dim

    def forward(self, results):
        # boxes in format [score, label, x, y, z, x, y, z]
        result = results.clone().detach()
        result = result[result[:,0].argsort(dim=0, descending=True)] # sort by score 
        result = result[result[:,0] >= self.score_tr] # score tr
        
        if result.nelement() == 0:
            return result
        
        filtered = []
        while result.nelement() != 0:
            filtered.append(result[[0]])
            ious = y9.bbox_iou(result[0, 2:], result[:, 2:], iou_mode=True)
            if self.del_dup_class:
                result = result[torch.logical_and(ious.squeeze()<=self.iou_tr, result[:,1]!=result[0, 1])]
            else:
                result = result[ious.squeeze()<=self.iou_tr]

        filtered = torch.cat(filtered, dim=0)
        if self.normalize:
            filtered[2:] = filtered[2:]/ torch.tensor(self.series_dim*2, dtype=filter.dtype).to(filtered.device)[2,1,0,5,4,3]
        return filtered 
    
class Nms3dAxial(nn.Module):
    def __init__(self, iou_treshold=0.3, score_treshold = 0., del_dup_class=False, unique_cls = 2, split_lr=False, normalize=False, series_dim=None) -> None:
        super().__init__()
        self.score_tr = score_treshold
        self.iou_tr = iou_treshold
        self.del_dup_class = del_dup_class
        self.unique_cls = unique_cls
        self.split_lr = split_lr
        self.normalize = normalize
        self.series_dim = series_dim

    def forward(self, results):
        # boxes in format [score, label, x, y, z, x, y, z]
        aresult = results.clone().detach()
        out = []
        for i in range(self.unique_cls): # max two objects per unique class
            result = aresult[aresult[:,1]==i]
            result = result[result[:,0].argsort(dim=0, descending=True)] # sort by score 
            result = result[result[:,0] >= self.score_tr] # score tr
            
            if result.nelement() == 0:
                return result
            
            filtered = []
            while result.nelement() != 0:
                filtered.append(result[[0]])
                ious = y9.bbox_iou(result[0, 2:2+6], result[:, 2:2+6], iou_mode=True)
                if self.del_dup_class:
                    result = result[torch.logical_and(ious.squeeze()<=self.iou_tr, result[:,1]!=result[0, 1])]
                else:
                    result = result[ious.squeeze()<=self.iou_tr]

            result = torch.cat(filtered, dim=0)
            out.append(result[0:2]) #limit to max 2 foraminas per image

        filtered =  torch.cat(out, dim=0)
        filtered = filtered.clamp(min=torch.tensor([0,0,0,0,0,0,0,0], dtype=filtered.dtype).to(filtered.device))
        if self.normalize:
            filtered[2:] = filtered[2:]/ torch.tensor(self.series_dim*2, dtype=filter.dtype).to(filtered.device)[2,1,0,5,4,3]
        return filtered 
    
def lineinter(line1, line2):
    inter = (torch.min(line1[5], line2[:,5]) - torch.max(line1[2], line2[:,2])).clamp(0)
    uni = (torch.max(line1[5], line2[:,5]) - torch.min(line1[2], line2[:,2])).clamp(0) - inter
    return inter/(uni + 1e-7)

class Nms3dSagittalForamina(nn.Module):
    def __init__(self, iou_treshold=0.3, score_treshold = 0., del_same_depth=False, unique_cls = 2) -> None:
        super().__init__()
        self.score_tr = score_treshold
        self.iou_tr = iou_treshold
        self.del_same_depth = del_same_depth
        self.unique_cls = unique_cls

    def forward(self, results):
        # boxes in format [score, label, x, y, z, x, y, z]
        aresult = results.clone().detach()
        out = []
        for i in range(self.unique_cls): # max two objects per unique class
            result = aresult[aresult[:,1]==i]
            result = result[result[:,0].argsort(dim=0, descending=True)] # sort by score 
            result = result[result[:,0] >= self.score_tr] # score tr
            
            if result.nelement() == 0:
                return result
            
            filtered = []
            while result.nelement() != 0:
                filtered.append(result[[0]])
                ious = y9.bbox_iou(result[0, 2:2+6], result[:, 2:2+6], iou_mode=True)
                if self.del_same_depth:
                    li = lineinter(result[0, 2:2+6], result[:, 2:2+6])
                    result = result[torch.logical_and(ious.squeeze()<=self.iou_tr, li < 0.1)]
                else:
                    result = result[ious.squeeze()<=self.iou_tr]

            result = torch.cat(filtered, dim=0)
            out.append(result[0:2]) #limit to max 2 foraminas per image

        return torch.cat(out, dim=0)
    
class BoxModel(nn.Module):
    def __init__(self, backbone_name:str = 'resnet_18', series_dim:list[int]=[128]*3, num_classes:int=1, device=None,
                 use_features:Union[str, list[int]]=[0], reg_max:int=16, pretrained:bool=False, postprocess=None):
        # backbone_name - timm model to use as backbone
        # series_dim - dimentionality of the series [num_channels, im_width, im_height]
        # use_features - features to use from backbone output:
        #        example model outputs features with dim [64, 64, 128, 256, 512] 
        #               'last' or [0] will take only last layer
        #               'all' will take all layers
        #               [0, 1, 2] will take last three layers

        super().__init__()
        # model outputs bounding box in yolo format [x_mid, y_mid, width, height] (normalized)
        # chose feature extractor from timm models
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.num_classes = num_classes
        self.series_dim = series_dim

        opts = {
        'model': 'resnet',
        'device': self.device,
        'phase': 'train',
        }
        self.backbone_name = backbone_name

        model_pretrained_params = {
            'resnet_10': {'model_depth': 10, 'resnet_shortcut': 'B'},
            'resnet_10_23dataset': {'model_depth': 10, 'resnet_shortcut': 'B'},
            'resnet_18': {'model_depth': 18, 'resnet_shortcut': 'A'},
            'resnet_18_23dataset': {'model_depth': 18, 'resnet_shortcut': 'A'},
            'resnet_34': {'model_depth': 34, 'resnet_shortcut': 'A'},
            'resnet_34_23dataset': {'model_depth': 34, 'resnet_shortcut': 'A'}
        }
        for model_name, model_dict in model_pretrained_params.items():
            model_pretrained_params[model_name] = Struct({**model_dict, **opts})
        
        def construct_network(feature_extractor, model_pretrained_params):
            model = MedNet(feature_extractor, model_pretrained_params)
            return model

        self.feature_extractor = construct_network(backbone_name, model_pretrained_params).to(self.device)

        if pretrained:
            self.feature_extractor.init_FE(self.device)

        self.all_channels = [64, 128, 256, 512]
        self.reduction = [4, 8, 8, 8]

        # features dimention
        if use_features=='all':
            self.in_channels = self.all_channels
            self.featmap_stride = self.reduction
            self.fl = list(range(len(self.all_channels)))
        elif use_features=='last':
            self.in_channels = [self.all_channels[-1]]
            self.featmap_stride = [self.reduction[-1]]
            self.fl = [len(self.all_channels)-1]
        elif type(use_features)==list:
            self.in_channels = [self.all_channels[len(self.all_channels)-(1+i)] for i in sorted(use_features, reverse=True)]
            self.featmap_stride = [self.reduction[len(self.reduction)-(1+i)] for i in sorted(use_features, reverse=True)]
            self.fl= [len(self.all_channels)-(1+i) for i in sorted(use_features, reverse=True)]

        if postprocess:
            self.postprocess = postprocess
        else:
            self.postprocess = Nms3d(0.3, 0.2, True)

        self.head = y9.Detect3d(nc=num_classes, ch=self.in_channels, strides=self.featmap_stride, reg_max=reg_max).to(self.device)
        
        h = {
            "device": self.device,
            "cls_pw":None,
            "label_smoothing": 0.0,
            "fl_gamma": 0.0
        }
        self.CL = y9.ComputeLoss(self.head, h)

    
    def forward(self, x: torch.Tensor) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tensor): input series
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
    
        x = self.feature_extractor(x.unsqueeze(1))
        x = [x[i] for i in self.fl]

        return self.head(x)

    def get_loss(self, series, gt_boxes, gt_labels):
        head_out = self.forward(series)  
        loss, loss_split, pred_a_boxes = self.CL(head_out,
                                        gt_boxes, gt_labels)
        metrics = dict(loss_cls=loss_split[1], loss_bbox=loss_split[0], loss_dfl=loss_split[2])

        return loss, metrics

    def predict(self, series):
        scores, labels, dbox = self.forward(series)
        result_list = []
        for score, label, boxes in zip(scores, labels, dbox):
            result = self.postprocess(torch.cat((score, label, boxes), dim=0).permute(1,0))
            result_list.append({'boxes': result[:, 2:], 'scores': result[:, 0], 'labels': result[:, 1].int()})
        return result_list