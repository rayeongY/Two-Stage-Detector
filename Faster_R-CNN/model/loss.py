

import torch.nn as nn


class RPNLoss(nn.module):
    def __init__(self):
        super(RPNLoss, self).__init__()
        
        self.cross_entropy = nn.CrossEntropyLoss
        self.smooth_l1 = nn.SmoothL1Loss

    
    def forward(
        self,
        pred,
        target,
        logger,
        n_iter
    ):
        
        rpn_object_loss = self.cross_entropy(pred, target)
        rpn_coordi_loss = self.smooth_l1(pred, target)

        logger.add_scalar('#train #objectLoss', rpn_object_loss.item(), n_iter)
        logger.add_scalar('#train #coordinatesLoss', rpn_coordi_loss.item(), n_iter)

        return rpn_object_loss, rpn_coordi_loss



class ClassificationLoss(nn.module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()


    def forward(
        pred,
        target,
        logger,
        n_iter
    ):
        loss = pred - target


        return loss