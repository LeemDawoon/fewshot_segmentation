import torch
import torch.nn.functional as F


def binary_cross_entropy(input, target, weight=None, size_average=True):
    loss = F.binary_cross_entropy(input, target, weight=weight, size_average=size_average)
    return loss


def cross_entropy(input, target, weight=None, reduction='mean'):
    # print('cross_entropy weight: ', weight)
    loss = F.cross_entropy(input, target, weight=weight, reduction=reduction)
    return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input, target, weight=weight, size_average=size_average, ignore_index=250)
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(input.device)

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, target, weight=weight, reduce=False, size_average=False, ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)


def dice(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def dice_disc_cup(input, target):
    n_class = input.size()[1]
    loss_list = []
    for c in range(n_class):
        smooth = 1.
        iflat = input[:, int(c), :, :].contiguous().view(-1)
        tflat = target[:, int(c), :, :].contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        loss_list.append(loss)
    if len(loss_list) != 2:
        print('loss_list', loss_list)
    return loss_list[0] * 0.4 + loss_list[1] * 0.6