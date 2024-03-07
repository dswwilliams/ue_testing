import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import sys
sys.path.append("../")
from ue_testing.utils.prototype_utils import extract_prototypes, segment_via_prototypes
from ue_testing.utils.disk_utils import load_checkpoint_if_exists
from ue_testing.utils.downsampling_utils import ClassWeightedModalDownSampler, downsample_labels
from ue_testing.models.base_model import BaseModel

class GammaSSLModel(BaseModel):
    """
    Model class for GammaSSL-style models.
    This includes both vanilla seg nets, as well as those using prototype segmentation.
    Primarily provides initialisation of pretrained segmentation network, 
        as well as method for getting segmentation masks and per-pixel uncertainties.
    """
    def __init__(self, opt, known_class_list=None, training_dataset=None):
        super().__init__(opt, known_class_list)

        self.seg_net, self.patch_size = self.init_seg_net()
        self.dataset_prototypes = self.init_dataset_prototypes()
        self.class_weighted_modal_downsampler = ClassWeightedModalDownSampler(self.known_class_list)
        self.init_prototype_dataset(training_dataset)

        self.batch_prototypes = None
        self.old_prototypes = None

    def init_seg_net(self):
        """ Initialise segmentation network and load checkpoint. """
        # determine model architecture
        if self.opt.model_arch == "vit_m2f":
            from models.seg_nets.vit_m2f_seg_net import ViT_M2F_SegNet as SegNet
            self.crop_size = 224
        elif self.opt.model_arch == "deeplab":
            from models.seg_nets.deeplab_seg_net import DeepLabSegNet as SegNet
            self.crop_size = 256
            
        # instantiate segmentation network
        seg_net = SegNet(self.device, self.opt, num_known_classes=self.num_known_classes)
        torch.save(seg_net.state_dict(), "vit_seg_net.pth")
        if self.opt.lora_rank is not None:
            import loralib as lora
            lora.mark_only_lora_as_trainable(seg_net.encoder)

        # set to eval mode
        seg_net.eval()

        # get patch_size if it exists
        if hasattr(seg_net.encoder, "patch_size"):
            patch_size = seg_net.encoder.patch_size
        else:
            patch_size = None

        # load in model weights from training
        load_checkpoint_if_exists(seg_net, self.opt.save_path)

        return seg_net, patch_size
    
    def get_val_seg_masks(self, imgs):
        """
        Get segmentation masks and uncertainty estimates from query and target branches for validation.

        Args:
            imgs: input images of shape [batch_size, 3, H, W]

        Returns:
            dict: {"query": query, "target": target}
                where query and target are dicts containing:
                    segs: segmentation masks of shape [batch_size, H, W]
                    uncertainty_maps: uncertainty maps of shape [batch_size, H, W]
        """
        if self.opt.use_proto_seg:
            seg_masks_K_q = self.proto_segment_imgs(imgs, use_dataset_prototypes=True)
        else:
            seg_masks_K_q = self.get_seg_masks(imgs, high_res=True, branch="query")
        segs_K_q = torch.argmax(seg_masks_K_q, dim=1)
        ms_imgs_q = torch.max(seg_masks_K_q, dim=1)[0]
        uncertainty_maps_q = 1 - ms_imgs_q
        query = {"segs": segs_K_q, "uncertainty_maps": uncertainty_maps_q}

        return {"query":query}
    
    def get_seg_masks(self, imgs, high_res=False, masks=None, branch=None):
        """
        Get segmentation masks from the input images, imgs.
        Determines the method based on branch arg and training options.

        Options:
        - branch = "query" and use_proto_seg = True
        - branch = "query" and use_proto_seg = False
        - branch = "target" and frozen_target = True
        - branch = "target" and frozen_target = False
        """

        if branch == "target" and self.target_seg_net:
            return self.target_seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks)
        elif branch == "target" and not self.target_seg_net:
            return self.seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks)
        elif branch == "query" and self.opt.use_proto_seg:
            return self.proto_segment_imgs(imgs, use_dataset_prototypes=False, output_spread=False, include_void=False, masks=masks)
        elif branch == "query" and not self.opt.use_proto_seg:
            return self.seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks)
        else:
            return self.seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks)


    def init_dataset_prototypes(self):
        """ Load dataset prototypes if they exist. """
        if self.opt.prototypes_path is not None:
            print("loading prototypes from ->", self.opt.prototypes_path)
            if self.opt.prototypes_path[-4:] == ".pkl":
                from ue_testing.utils.prototype_utils import load_prototypes_from_pkl
                self.dataset_prototypes = load_prototypes_from_pkl(self.opt.prototypes_path, self.device)
            else:
                checkpoint = torch.load(self.opt.prototypes_path, map_location=self.device)
                self.dataset_prototypes = checkpoint["prototypes"]
        else:
            self.dataset_prototypes = None

    def init_prototype_dataset(self, proto_dataset):
        """ Initialise dataset used to calculate dataset prototypes. """
        if proto_dataset is None:
            from datasets.cityscapes_bdd_dataset import CityscapesxBDDDataset
            self.proto_dataset = CityscapesxBDDDataset(
                                        labelled_dataroot=self.opt.cityscapes_dataroot, 
                                        bdd_dataroot=self.opt.unlabelled_dataroot, 
                                        no_appearance_transform=True,
                                        min_crop_ratio=1,
                                        max_crop_ratio=1,
                                        add_resize_noise=False,
                                        only_labelled=True,     # NOTE: only reading in labelled data
                                        use_imagenet_norm=self.opt.use_imagenet_norm,
                                        no_colour=True,
                                        crop_size=self.crop_size,
                                        )
        
    def proto_segment_features(self, features, img_spatial_dims=None, use_dataset_prototypes=False, include_void=False):
        """
        Get segmentation masks from input features by calculating similarity w.r.t. class prototypes.

        Args:
            features: input features of shape [batch_size, feature_length, H, W]
            img_spatial_dims: spatial dimensions of input images
            use_dataset_prototypes: if True, use dataset prototypes
                i.e. those calculated from entire dataset
            include_void: if True, use gamma to get p(unknown)

        Returns:
            seg_masks: segmentation masks of shape [batch_size, num_classes, H, W]
        """

        # device between prototypes calculated from batch or entire dataset
        if use_dataset_prototypes:
            prototypes = self.dataset_prototypes
        else:
            prototypes = self.batch_prototypes
        
        # get projected features
        proj_features = self.seg_net.projection_net(features)

        if include_void:
            _gamma = self.gamma
        else:
            _gamma = None

        # calculate segmentation masks from projected features and prototypes
        seg_masks = segment_via_prototypes(
                                    proj_features,
                                    prototypes.detach(),         # NOTE: to prevent backprop
                                    gamma=_gamma,
                                    )
        if img_spatial_dims is not None:
            H,W = img_spatial_dims
            seg_masks = F.interpolate(seg_masks, size=(H,W), mode="bilinear", align_corners=True)
        return seg_masks

    def proto_segment_imgs(self, imgs, use_dataset_prototypes=False, include_void=False, masks=None):
        """
        Calculate segmentation masks from input images by extracting features, then using prototypes.

        Args:
            imgs: input images of shape [batch_size, 3, H, W]
            use_dataset_prototypes: if True, use dataset prototypes
                i.e. those calculated from entire dataset
            include_void: if True, use gamma to get p(unknown)

        Returns:
            seg_masks: segmentation masks of shape [batch_size, num_classes, H, W]
        """
        features = self.seg_net.extract_features(imgs, masks=masks)

        seg_masks = self.proto_segment_features(
                        features, 
                        img_spatial_dims=imgs.shape[2:], 
                        use_dataset_prototypes=use_dataset_prototypes, 
                        include_void=include_void)
        return seg_masks
    
    @torch.no_grad()
    def calculate_dataset_prototypes(self):
        """
        Calculate prototypes from entire dataset of labelled data.
        Batches of labelled data are obtained from the prototype dataloader.
        """
        if self.opt.prototypes_path is None:
            dataloader = torch.utils.data.DataLoader(
                                    self.proto_dataset, 
                                    batch_size=self.opt.batch_size, 
                                    shuffle=False, 
                                    num_workers=self.opt.num_workers, 
                                    drop_last=True)
            
            iterator = tqdm(dataloader)
            print("calculating dataset prototypes...")
            prototypes_sum = 0
            for labelled_dict,_ in iterator:
                labelled_imgs = labelled_dict["img"].to(self.device)
                labels = labelled_dict["label"].to(self.device) 

                # extract features
                labelled_features = self.seg_net.extract_proj_features(labelled_imgs)

                # downsample labels to match feature spatial dimensions
                low_res_labels = downsample_labels(
                                    features=labelled_features, 
                                    labels=labels, 
                                    downsampler=self.class_weighted_modal_downsampler,
                                    )

                # calculate prototypes
                prototypes = extract_prototypes(labelled_features, low_res_labels, output_metrics=False)

                prototypes_sum += prototypes

            prototypes = F.normalize(prototypes_sum, dim=0, p=2)          # shape: [feature_length, num_known_classes]

            self.dataset_prototypes = prototypes
            return self.dataset_prototypes
        else:
            return None