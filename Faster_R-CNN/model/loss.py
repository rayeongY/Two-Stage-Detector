

import torch.nn as nn


class RPNLoss(nn.Module):
    def __init__(self):
        super(RPNLoss, self).__init__()
        
        self.cross_entropy = nn.CrossEntropyLoss
        self.smooth_l1 = nn.SmoothL1Loss

    
    def forward(
        self,
        pred,       # (BATCH_SIZE, f_w, f_h, 9, 2 + 4)
        target,     # (BATCH_SIZE, f_w, f_h, 9, 2 + 4)
        logger,
        n_iter
    ):
        
        is_object_mask = target[..., 0] > 0

        rpn_object_loss = self.cross_entropy(pred[..., :1], target[..., :1])
        rpn_coordi_loss = self.smooth_l1(pred[..., 2:][is_object_mask], target[..., 2:][is_object_mask])

        logger.add_scalar('#train #objectLoss', rpn_object_loss.item(), n_iter)
        logger.add_scalar('#train #coordinatesLoss', rpn_coordi_loss.item(), n_iter)

        _lambda = 10
        rpn_total_loss = rpn_object_loss + _lambda * rpn_coordi_loss
        return rpn_total_loss



class ClassificationLoss(nn.Module):        ## Fast R-CNN classification loss
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