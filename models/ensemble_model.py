import torch
import torch.nn.functional as F
import copy
import sys
sys.path.append("../")
from models.base_model import BaseModel
from utils.uncertainty_utils import calculate_predictive_entropy, calculate_mutual_information

class EnsembleModel(BaseModel):
    """
    Model class for ensemble models.
    Primarily provides initialisation of pretrained ensemble of segmentation networks, 
        as well as method for getting segmentation masks and per-pixel uncertainties.
    """
    def __init__(self, opt, known_class_list=None):
        super().__init__(opt, known_class_list)
        self.ensemble, self.patch_size = self.init_seg_net()


    def load_ensemble_from_checkpoints(self, ensemble_save_path, seg_net):
        """
        Creating ensemble as a list of seg nets.
        Checkpoint locations are read from a .txt file, found at ensemble_save_path.
        """
        # read in path to each ensemble checkpoint
        try:
            with open(ensemble_save_path) as f:
                save_paths = f.readlines()
        except FileNotFoundError:
            print(f"No ensemble save path found at: {ensemble_save_path}")
            sys.exit()

        # create ensemble as list of models
        model_list = []
        for save_path in save_paths:
            save_path = save_path.strip()
            checkpoint = torch.load(save_path, map_location="cpu")
            print("Weights loaded from -> ", save_path)
            pretrained_model = copy.deepcopy(seg_net)
            pretrained_model.load_state_dict(checkpoint, strict=True)
            model_list.append(pretrained_model)

    
        return model_list

    def init_seg_net(self):
        """ Initialise list of seg nets as ensemble."""
        # determine model architecture
        if self.opt.model_arch == "vit_m2f":
            from models.seg_nets.vit_m2f_seg_net import ViT_M2F_SegNet as SegNet
        elif self.opt.model_arch == "deeplab":
            from models.seg_nets.deeplab_seg_net import DeepLabSegNet as SegNet
            
        # instantiate segmentation network
        seg_net = SegNet(self.device, self.opt, num_known_classes=self.num_known_classes)
        if self.opt.lora_rank is not None:
            import loralib as lora
            lora.mark_only_lora_as_trainable(seg_net.encoder)

        # get patch_size if it exists
        if hasattr(seg_net.encoder, "patch_size"):
            patch_size = seg_net.encoder.patch_size
        else:
            patch_size = None

        # set to eval mode
        seg_net.eval()

        # load in model weights to copies of seg_net
        ensemble = self.load_ensemble_from_checkpoints(
                                ensemble_save_path=self.opt.save_path, 
                                seg_net=seg_net
                                )


        return ensemble, patch_size
    

    def get_val_seg_masks(self, imgs):
        if self.opt.uncertainty_metric == "pe":
            segs, uncertainty_maps = self.get_predictions_via_predictive_entropy(imgs)
        elif self.opt.uncertainty_metric == "mi":
            segs, uncertainty_maps = self.get_predictions_via_mutual_information(imgs)

        return {self.opt.uncertainty_metric: {"segs": segs, "uncertainty_maps": uncertainty_maps}}


    def get_predictions_via_predictive_entropy(self, imgs):
        """
        Get segmentations and uncertainty maps as the predictive entropy.
        Firstly, get the predictions as the mean of N forward passes through the ensemble.
        Each ensemble member is loaded to the gpu in turn, and then returned to the cpu.
        """
        for n in range(len(self.ensemble)):
            # load model to gpu
            model = self.ensemble[n].to(self.device)
            # get output from model
            output = model(imgs).detach()
            # take model off gpu
            model = model.to("cpu")
            output = F.interpolate(output, size=(imgs.shape[2], imgs.shape[3]), mode="bilinear", align_corners=True)
            if n == 0:
                mean_seg_net_predictions = output
            else:
                mean_seg_net_predictions += output
            del output

        mean_seg_net_predictions = mean_seg_net_predictions / len(self.ensemble)
        
        # getting segmentations (calculate mean p(class) over the forward passes then take argmax to determine for which class it was highest)
        segs = torch.argmax(mean_seg_net_predictions, dim=1)
        mean_seg_net_predictions = torch.softmax(mean_seg_net_predictions, dim=1)

        # calculate uncertainty measure
        predictive_entropy = calculate_predictive_entropy(mean_seg_net_predictions)

        return segs, predictive_entropy


    def get_predictions_via_mutual_information(self, imgs):
        """
        Get segmentations and uncertainty maps as the mutual information.
        Firstly, get the predictions as N batches of seg masks for the N ensemble members.
        Each ensemble member is loaded to the gpu in turn, and then returned to the cpu.
        """

        # get predictions as N batches of seg masks for the N ensemble members
        # loading members to gpu in turn
        seg_net_predictions = []
        for n in range(len(self.ensemble)):
            # load model to gpu
            model = self.ensemble[n].to(self.device)
            # get output from model
            output = model(imgs).detach()
            # take model off gpu
            model = model.to("cpu")
            output = F.interpolate(output, size=(imgs.shape[2], imgs.shape[3]), mode="bilinear", align_corners=True)
            seg_net_predictions.append(output.unsqueeze(0))
            del output
        
        seg_net_predictions = torch.cat(seg_net_predictions, dim=0)

        # getting segmentations (calculate mean p(class) over the forward passes then take argmax to determine for which class it was highest)
        segs = torch.argmax(seg_net_predictions.mean(0), dim=1)
        seg_net_predictions = torch.softmax(seg_net_predictions, dim=2)

        # calculate uncertainty measure
        mutual_information = calculate_mutual_information(seg_net_predictions)

        return segs, mutual_information