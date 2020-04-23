import logging

# from core.augmentations.custom_augmentations import *
import torch
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    CenterCrop,
    ColorJitter,
    FiveCrop,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    Scale,
    TenCrop,
    ToTensor,
    Normalize,
)

from core.augmentations.custom_np_augmentations import (
    NumpyScale,
    NumpyCenterCrop,
    NumpyRandomCrop,
    NumpyRandomZCrop,
    NumpyRandomRotate,
    NumpyRandomHorizontalFlip,
    NumpyRandomVerticalFlip,
    TransposeInput,
    TransposeOutput,
    NumpyNormalize,
)

logger = logging.getLogger("doai")
key2aug = {
    'np_scale': NumpyScale,
    'np_center_crop': NumpyCenterCrop,
    'np_random_crop': NumpyRandomCrop,
    'np_random_z_crop': NumpyRandomZCrop,
    'np_random_rotate': NumpyRandomRotate,
    'np_random_hflip': NumpyRandomHorizontalFlip,
    'np_random_vflip': NumpyRandomVerticalFlip,
    'np_transpose_input': TransposeInput,
    'np_transpose_output': NumpyRandomVerticalFlip,
    'np_normalize': NumpyNormalize,
    # custom augmentations
    # "custom_gamma": custom_augmentations.AdjustGamma,
    # "custom_hue": custom_augmentations.AdjustHue,
    # "custom_brightness": custom_augmentations.AdjustBrightness,
    # "custom_saturation": custom_augmentations.AdjustSaturation,
    # "custom_contrast": custom_augmentations.AdjustContrast,
    # "custom_random_crop": custom_augmentations.RandomCrop,
    # "custom_random_hflip": custom_augmentations.RandomHorizontallyFlip,
    # "custom_random_vflip": custom_augmentations.RandomVerticallyFlip,
    # "custom_scale": custom_augmentations.Scale,
    # "custom_random_crop_resize": custom_augmentations.RandomCropResize,
    # "custom_random_resize_crop": custom_augmentations.RandomResizeCrop,
    # "custom_rotate": custom_augmentations.RandomRotate,
    # "custom_translate": custom_augmentations.RandomTranslate,
    # "custom_center_crop": custom_augmentations.CenterCrop,

    # pytorch augmentations
    "center_crop": CenterCrop,
    "color_jitter": ColorJitter,
    "five_crop": FiveCrop,
    "random_affine": RandomAffine,
    "random_crop": RandomCrop,
    "random_hflip": RandomHorizontalFlip,
    "random_vflip": RandomVerticalFlip,
    "random_resized_crop": RandomResizedCrop,
    "random_rotation": RandomRotation,
    "resize": Resize,
    "scale": Scale,
    "ten_crop": TenCrop,
    "to_tensor": ToTensor,
    # "n_crop_to_tensor": transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    "normalize": Normalize,
    # "n_crop_normalize": transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
}


# def get_composed_custom_augmentations(aug_dict):
#     if aug_dict is None:
#         logger.info("Using No Custom Augmentations")
#         return None
#
#     augmentations = []
#     for aug_key, aug_param in aug_dict.items():
#         if aug_key == 'color_jitter' or aug_key == 'random_affine' or aug_key == 'normalize':
#             augmentations.append(key2aug[aug_key](*aug_param))
#         else:
#             augmentations.append(key2aug[aug_key](aug_param))
#         logger.info("Using {} custom aug with params {}".format(aug_key, aug_param))
#     return custom_augmentations.Compose(augmentations)

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        if aug_key in ['color_jitter', 'normalize', 'np_normalize']:
            augmentations.append(key2aug[aug_key](*aug_param))
        elif aug_key in ['random_affine', 'random_resized_crop']:
            augmentations.append(key2aug[aug_key](*aug_param))
        elif aug_key == 'n_crop_normalize':
            normalize = key2aug['normalize'](*aug_param)
            augmentations.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        elif aug_key in ['to_tensor', 'np_transpose_input', 'np_transpose_output', 'np_random_vflip', 'np_random_hflip']:
            augmentations.append(key2aug[aug_key]())
        elif aug_key == 'n_crop_to_tensor':
            augmentations.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        else:
            augmentations.append(key2aug[aug_key](aug_param))

        logger.info("Using {} aug with params {}".format(aug_key, aug_param))

    return Compose(augmentations)

