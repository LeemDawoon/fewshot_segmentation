"""
Misc Utility functions
"""
import os
import logging
import numpy as np
import yaml

from collections import OrderedDict

logger = logging.getLogger("doai")

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove `module.` used in DataParallel
        if k[:7] == 'module.':
            name = k[7:]
        else:
            name = k
        name = name.replace('conv.1', 'conv1').replace('conv.2', 'conv2')
        name = name.replace('norm.1', 'norm1').replace('norm.2', 'norm2')

        new_state_dict[name] = v
    return new_state_dict
    
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor( yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    return yaml.load(stream, OrderedLoader)

