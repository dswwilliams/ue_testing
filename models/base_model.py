import torch
import torch.nn as nn
import sys
sys.path.append("../")
from ue_testing.utils.device_utils import init_device


class BaseModel(nn.Module):
    """
    Base class for all models.
    Need to implement init_seg_net and get_val_seg_masks.
    i.e. how to init the segmentation network and how to get segmentation masks and per-pixel uncertainties for testing.
    """
    def __init__(self, opt, known_class_list):
        super().__init__()
        if known_class_list is None:
            # use cityscapes as default
            self.known_class_list = ["road", "sidewalk", "building", "wall", "fence", "pole", 
                                "traffic_light", "traffic_sign", "vegetation", "terrain", 
                                "sky", "person", "rider", "car", "truck", "bus", "train", 
                                "motorcycle", "bicycle"]
        else:
            self.known_class_list = known_class_list

        self.opt = opt
        self.num_known_classes = len(self.known_class_list)
        self.device = init_device(gpu_no=self.opt.gpu_no, use_cpu=self.opt.use_cpu)

    def init_seg_net(self):
        """ Initialise segmentation network and load checkpoint if it exists. """
        return NotImplementedError

    def get_val_seg_masks(self, imgs):
        """ Get segmentation masks and uncertainty estimates for testing. """
        return NotImplementedError
    

