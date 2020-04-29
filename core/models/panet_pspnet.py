"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from .vgg_pspnet import Encoder


class PANetPSPNet(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(
        self, 
        in_channels=3, 
        n_classes=None,
        pretrained_path=None, 
        pretrained=True,
        align=True,
        cfg=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained

        self.config = cfg or {'align': False}
        self.align = align

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([('backbone', Encoder(in_channels, self.pretrained_path)),]))
        # summary(Encoder(in_channels, self.pretrained_path) , (3, 224, 224))


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                # 1, 5, 3, 224, 224
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        # print('>>> panet supp_imgs', supp_imgs.shape)
        # print('>>> panet fore_mask', fore_mask.shape)
        # print('>>> panet back_mask', back_mask.shape)
        # print('>>> panet qry_imgs', qry_imgs.shape)
        # >>> panet supp_imgs torch.Size([1, 5, 3, 224, 224])
        # >>> panet fore_mask torch.Size([1, 5, 1, 224, 224])
        # >>> panet back_mask torch.Size([1, 5, 1, 224, 224])
        # >>> panet qry_imgs torch.Size([1, 3, 224, 224])

        n_ways = supp_imgs.shape[0]
        n_shots = supp_imgs.shape[1]
        n_queries = qry_imgs.shape[0]
        # batch_size = supp_imgs[0][0].shape[0]
        batch_size = 1
        img_size = supp_imgs.shape[-2:]
        ###### Extract features ######
        # imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0),], dim=0)
        imgs_concat = torch.cat([
            supp_imgs.view(n_ways*n_shots, self.in_channels, *img_size),
            qry_imgs
        ], dim=0)

        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size, :, :, :].view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:, :, :, :].view(n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]]) for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]]) for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.align and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        # print('fts.shape', fts.shape)
        # print('prototype.shape', prototype.shape)
        # print('dist.shape', dist.shape)
        # fts.shape torch.Size([1, 64, 224, 224])
        # prototype.shape torch.Size([1, 64])
        # dist.shape torch.Size([1, 224, 224])
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        # masked average pooling
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear') # resize equal to mask size
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss
