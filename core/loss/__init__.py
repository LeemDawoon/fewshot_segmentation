import logging
import functools

from core.loss.loss import (
    binary_cross_entropy,
    cross_entropy,
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
    dice,
    dice_disc_cup,
)


logger = logging.getLogger("doai")

key2loss = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "cross_entropy2d": cross_entropy2d,
    "bootstrapped_cross_entropy2d": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy2d": multi_scale_cross_entropy2d,
    "dice": dice,
    "dice_disc_cup": dice_disc_cup,
}


def get_loss_function(cfg):
    if cfg["train"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg["train"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
