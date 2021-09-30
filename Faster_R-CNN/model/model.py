from common.utils import *

import timm

import torch
import torch.nn as nn
from torchvision.models import vgg16


class FasterRCNN(nn.module):
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
        self.rpn = RPN()

        ## RoI Pooling ???
        self.pooling = RoIPool(
        
        )

        ## Classifier
        self.classifier = FastRCNN(

        )


        if weight_path is not None:
            self.load_weights(weight_path)


    def forward(self, x):

        anchors = anchor_generator(x)
        feature_maps = self.backbone(x)

        if self.training:
            t_anchors = t_anchor_generator(anchors, x)
            score_and_regs = self.rpn(feature_maps)

            proposals = self.proposal(score_and_regs, anchors)
            t_proposals = t_prop_generator(proposals)

            rois = self.pooling(proposals)
            preds = self.classifier(rois)

            return preds
        
        else:
            score_and_regs = self.rpn(feature_maps)
            proposals = self.proposal(score_and_regs, anchors)
            rois = self.pooling(proposals)
            preds = self.classifier(rois)

            ## NMS
            result = nms(preds)
            
            return result


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


class RPN(nn.modeul):
    def __init__(
        self,
    ):
        super(RPN, self).__init__()


class RoIPool(nn.modeul):
    def __init__(
        self,
    ):
        super(RoIPool, self).__init__()


class FastRCNN(nn.modeul):
    def __init__(
        self,
    ):
        super(FastRCNN, self).__init__()


