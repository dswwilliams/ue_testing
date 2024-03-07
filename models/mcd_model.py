import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import sys
import segmentation_models_pytorch as smp
sys.path.append("../")
from models.base_model import BaseModel
from torchvision.models.resnet import BasicBlock
from models.seg_nets.smp_mcd_resnet import MCDResNetEncoder
from utils.disk_utils import load_checkpoint_if_exists
from utils.uncertainty_utils import calculate_predictive_entropy, calculate_mutual_information


class MCDModel(BaseModel):
    """
    Model class for Monte Carlo Dropout (MCD) models.
    Primarily provides initialisation of pretrained MCD segmentation network, 
        as well as method for getting segmentation masks and per-pixel uncertainties.
    """
    def __init__(self, opt, known_class_list=None):
        super().__init__(opt, known_class_list)

        self.seg_net, self.patch_size = self.init_seg_net()

    def init_seg_net(self):
        """ Initialise segmentation network and load checkpoint. """
        # determine model architecture
        if self.opt.model_arch == "vit_m2f":
            seg_net = self.get_mcd_vit_seg_net()
        elif self.opt.model_arch == "deeplab":
            seg_net = self.get_mcd_deeplab_seg_net()

        # get patch_size if it exists
        if hasattr(seg_net.encoder, "patch_size"):
            patch_size = seg_net.encoder.patch_size
        else:
            patch_size = None

        # load in model weights from training
        load_checkpoint_if_exists(seg_net, self.opt.save_path)

        return seg_net, patch_size


    def get_mcd_vit_seg_net(self):
        from models.seg_nets.vit_m2f_seg_net import ViT_M2F_SegNet

        # making sure we get mcd vit encoder
        if "mcd" not in self.opt.vit_type:
            self.opt.vit_type += "_mcd"

        seg_net = ViT_M2F_SegNet(device=self.device, opt=self.opt, num_known_classes=self.num_known_classes)
        # all modules of model to eval, except dropout
        seg_net.eval()
        for module in seg_net.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        return seg_net


    def get_mcd_deeplab_seg_net(self):
        smp.encoders.encoders["mcd_resnet18"] = {
        "encoder": MCDResNetEncoder, # encoder class here
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
            "dropout_prob": self.opt.dropout_prob,
        },
        }
        seg_net = smp.DeepLabV3Plus(encoder_name="mcd_resnet18", 
                                                in_channels=3, 
                                                classes=self.num_known_classes, 
                                                encoder_weights=None,
                                                ).to(self.device)
        seg_net.eval()
        return seg_net
    

    def get_val_seg_masks(self, imgs):
        if self.opt.uncertainty_metric == "pe":
            segs, uncertainty_maps = self.get_predictions_via_predictive_entropy(imgs)
        elif self.opt.uncertainty_metric == "mi":
            segs, uncertainty_maps = self.get_predictions_via_mutual_information(imgs)

        return {self.opt.uncertainty_metric: {"segs": segs, "uncertainty_maps": uncertainty_maps}}


    def get_predictions_via_predictive_entropy(self, imgs):
        """
        Get segmentations and uncertainty maps as the predictive entropy.
        Firstly, get the predictions as the mean of N forward passes.
        """
        for n in range(self.opt.n_mcd_samples):
            seg_masks = self.seg_net(imgs).detach()
            seg_masks = F.interpolate(seg_masks, size=(imgs.shape[2], imgs.shape[3]), mode="bilinear", align_corners=True)
            if n == 0:
                mean_seg_net_predictions = seg_masks
            else:
                mean_seg_net_predictions += seg_masks
            del seg_masks

        mean_seg_net_predictions = mean_seg_net_predictions / self.opt.n_mcd_samples
        
        # getting segmentations (calculate mean p(class) over the forward passes 
        #   then take argmax to determine for which class it was highest)
        segs = torch.argmax(mean_seg_net_predictions, dim=1)
        mean_seg_net_predictions = torch.softmax(mean_seg_net_predictions, dim=1)

        # calculate uncertainty measure
        predictive_entropy = calculate_predictive_entropy(mean_seg_net_predictions)

        return segs, predictive_entropy


    def get_predictions_via_mutual_information(self, imgs):
        """
        Get segmentations and uncertainty maps as the mutual information.
        Firstly, get the predictions as N batches of seg masks for the N MCD samples.
        """
        seg_net_predictions = []
        for _ in range(self.opt.n_mcd_samples):
            seg_masks = self.seg_net(imgs).detach()
            seg_masks = F.interpolate(seg_masks, size=(imgs.shape[2], imgs.shape[3]), mode="bilinear", align_corners=True)
            seg_net_predictions.append(seg_masks.unsqueeze(0))
            del seg_masks
        
        seg_net_predictions = torch.cat(seg_net_predictions, dim=0)

        # getting segmentations (calculate mean p(class) over the forward passes 
        #   then take argmax to determine for which class it was highest)
        segs = torch.argmax(seg_net_predictions.mean(0), dim=1)
        seg_net_predictions = torch.softmax(seg_net_predictions, dim=2)

        # calculate uncertainty measure
        mutual_information = calculate_mutual_information(seg_net_predictions)

        return segs, mutual_information