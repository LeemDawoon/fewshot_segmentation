import copy
import torchvision.models as models
# pytorch provides alexnet, densenet, inception, resnet, squeezenet, and vgg

# segmentation models
from core.models.panet import PANet
from core.models.panet_unet import PANetUnet

# classification models


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    else:
        model = model(n_classes=n_classes, **param_dict)
        # model = model( **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "panet": PANet,
            "panet_unet": PANetUnet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
