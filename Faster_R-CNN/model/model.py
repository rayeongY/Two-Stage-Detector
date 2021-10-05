from numpy.lib.arraypad import pad
from common.utils import *
from loss import *

import timm

import torch
import torch.nn as nn
from torchvision.models import vgg16


class FasterRCNN(nn.Module):
    def __init__(
        self,
        model_opt,
        weight_path=None
    ):
        super(FasterRCNN, self).__init__()

        self.num_classes = model_opt["MODEL"]["NUM_CLASSES"]
        
        ## backbone: VGG16
        self.backbone = timm.create_model(
            model_name=model_opt["MODEL"]["BACKBONE"],
            pretrained=True,
            features_only=True
        )
        self.stage_channels = self.backbone.feature_info.channels()

        ## Region Proposal Network
        self.rpn = RPN(
            in_channels=...,
            model_opt=model_opt
        )

        ## RoI Pooling ???
        self.pooling = RoIPool(
        
        )

        ## Classifier
        self.classifier = FastRCNN(

        )

        
        self.rpnLoss = RPNLoss()
        self.clsLoss = ClassificationLoss()


        if weight_path is not None:
            self.load_weights(weight_path)


    def forward(self, x, logger, n_iter):

        anchors = anchor_generator(x)
        feature_maps = self.backbone(x)

        if self.training:
            t_anchors = self.t_anchor_generator(anchors, x)
            rpn_preds = self.rpn(feature_maps)

            proposals = self.create_proposals(rpn_preds, anchors)
            t_proposals = self.t_prop_generator(proposals)

            rois = self.pooling(proposals)
            preds = self.classifier(rois)

            rpn_loss = self.rpnLoss(rpn_preds, t_anchors, logger, n_iter)
            cls_loss = self.clsLoss(preds, t_proposals, logger, n_iter)

            return rpn_loss, cls_loss
        
        else:
            rpn_preds = self.rpn(feature_maps)
            proposals = self.proposal(rpn_preds, anchors)
            rois = self.pooling(proposals)
            preds = self.classifier(rois)

            ## NMS
            result = nms(preds)
            
            return result


    def create_proposals(
        self,
        rpn_preds,
        anchors,
    ):
        ## Transform the anchors according to the bounding box regression coefficients
        ## to generate transformed anchors
        ## -> then PRUNE the number of anchors by applying NMS
        ##    using the probability of an anchor being a foreground region

        proposals = None


        return proposals    ## proposals := ROIs & ROI scores



    def t_anchor_generator(
        self,
    ):
        ## GOAL: to PRODUCE 
        ##          - a set of "GOOD" anchors
        ##          - the corresponding fore/background labels and target regression codfficients
        ##       to train RPN.
        ##       
        ## ...
        ## (1) Eliminate the anchors being out of the image
        ## (2) Sampling positive/negative anchor boxes
        ##      - maybe positive anchors == foreground anchors? (whose overlap with some g.t. box is higher than a thresh.)
        ##      - background are whose overlap with any ground truth box is lower than a thresh.
        
        target_anchors = None

        ##


        return target_anchors



    def t_prop_generator(
        self,
    ):
        ## GOAL: to PRUNE the list of anchors produced by the proposal layer
        ##       and PRODUCE _class specific_ bounding box regression targets
        ##       that will be used to train FRCN
        ## ...
        ## (1) 

        target_proposals = None


        return target_proposals
    


    def load_weights(self, weight_path):
        
        print(f"load weights from : '{weight_path}'")
        with open(weight_path, "rb") as f:
            import numpy as np

            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i, (block, module) in enumerate(zip(self.module_cfgs, self.module_list)):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                print(conv_layer)
                if block["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]

                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_mean
                    )
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.running_var
                    )
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        conv_layer.bias
                    )
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

            print(
                f"{i} {block['type']:12s} load weights : [{ptr/1024*4/1024.:.3f}]/[{weights.size/1024*4/1024.:.3f}] mb",
                end="\n",
            )

        print(f"{ptr/1024*4/1024.:.0f}mb / {weights.size/1024*4/1024.:.0f}mb", end="\r")


class RPN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_opt
    ):
        super(RPN, self).__init__()

        self.model_opt = model_opt

        self.rpn_Conv = nn.Conv2d(          ## https://github.com/jwyang/faster-rcnn.pytorch/blob/pytorch-1.0/lib/model/rpn/rpn.py#:~:text=self.RPN_Conv%20%3D%20nn.Conv2d(self.din%2C%20512%2C%203%2C%201%2C%201%2C%20bias%3DTrue)
            in_channels, 
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        num_anchors = self.model_opt["MODEL"]["NUM_ANCHORS"]
        self.rpn_obj_score = nn.Conv2d(
            in_channels=512,
            out_channels=(2 * num_anchors),
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.rpn_coord_score = nn.Conv2d(
            in_channels=512,
            out_channels=(4 * num_anchors),
            kernel_size=3,
            stride=1,
            padding=0
        )


    def forward(self, f_map):
        x = self.rpn_Conv(f_map)

        rpn_obj_map = self.rpn_obj_score(x)
        rpn_coord_map = self.rpn_coord_score(x)

        return rpn_obj_map, rpn_coord_map




class RoIPool(nn.Module):
    def __init__(
        self,
    ):
        super(RoIPool, self).__init__()




class FastRCNN(nn.Module):
    def __init__(
        self,
    ):
        super(FastRCNN, self).__init__()


