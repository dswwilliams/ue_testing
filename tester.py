import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from collections import defaultdict
from test_utils import calculate_ue_metrics, calculate_miou, calculate_metrics_suite, plot_ue_metrics, update_running_totals
from device_utils import to_device
from datasets.val_datasets import ValDataset

class Tester():
    """
    Base class for all testers and validators

    - what should be included?
        - quantitative testing given a model and a dataset
        - qualitative testing given a model and a dataset

    - what about if there is a target and a query branch?
        - these will be different if the parameterisation of the two are different
        - this is true both when the target is frozen and when one branch is using prototype segmentation and the other is using a seg head
    - this functionality should be built on top of the base tester

    - what abstractions should be built into this base tester?
        - the model is defined as per the model class (see gammassl_public)
            - this will need to be included in this repo
        - also need to include datasets here
        - 

    """

    def __init__(self, opt, model):
        self.opt = opt
        self.model = model

        self._init_logging()

        self._init_device()
        self._init_val_datasets()

        # define known classes
        self.known_class_list = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
            "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        
        # define misclassification states
        self.states = {}
        self.states["tp"] = "n_accurate_and_certain"
        self.states["tn"] = "n_uncertain_and_inaccurate"
        self.states["fp"] = "n_inaccurate_and_certain"
        self.states["fn"] = "n_uncertain_and_accurate"

        # consider all classes by adding void to known classes (without changing original)
        self.class_labels = self.known_class_list.copy()
        if not self.class_labels[-1] == "void":
            self.class_labels.append("void")
        self.class_dict = {class_idx:class_label  for class_idx, class_label in enumerate(self.class_labels)}

        self.val_seg_idxs = {}

    def _init_logging(self):
        if self.opt.use_wandb:
            import wandb
            wandb.init(project=self.opt.wandb_project, config=self.opt)
        else:
            import visdom
            self.vis = visdom.Visdom(port=8097)


    def _init_val_datasets(self):
        self.val_seg_idxs = {}
        self.val_datasets = []
        cityscapes_val_dataset = ValDataset(
                                        name="CityscapesVal",
                                        dataroot=self.opt.cityscapes_dataroot, 
                                        use_imagenet_norm=self.opt.use_imagenet_norm, 
                                        val_transforms=self.opt.val_transforms,
                                        patch_size=self.model.patch_size,
                                        )
        self.val_datasets.append(cityscapes_val_dataset)
        self.val_seg_idxs[cityscapes_val_dataset.name] = np.random.choice(len(cityscapes_val_dataset), self.opt.n_val_segs, replace=False)

        bdd_val_dataset = ValDataset(
                            name="BDDVal",
                            dataroot=self.opt.bdd_val_dataroot, 
                            use_imagenet_norm=self.opt.use_imagenet_norm, 
                            val_transforms=self.opt.val_transforms,
                            patch_size=self.model.patch_size,
                            )
        self.val_datasets.append(bdd_val_dataset)
        self.val_seg_idxs[bdd_val_dataset.name] = np.random.choice(len(bdd_val_dataset), self.opt.n_val_segs, replace=False)

        for dataset in self.val_datasets:
            n_val_examples = len(dataset)
            print(dataset.name+" - Num. val examples", n_val_examples)


    def _init_dataloader(self, val_dataset):
        _batch_size = self.opt.val_batch_size if self.opt.val_batch_size is not None else self.opt.batch_size
        dataloader = torch.utils.data.DataLoader(
                                                    dataset=val_dataset, 
                                                    batch_size=_batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2, 
                                                    drop_last=False,
                                                    )
        return dataloader

    def _init_device(self):
        # if available (and not overwridden by opt.use_cpu) use GPU, else use CPU
        if torch.cuda.is_available() and self.opt.use_cpu == False:
            device_id = "cuda:" + self.opt.gpu_no
        else:
            device_id = "cpu"
        
        print("Device: ", device_id)
        self.device = torch.device(device_id)


    @torch.no_grad()
    def validate_batch(
                self,
                val_imgs, 
                val_labels,
                opt,
                ):

        outputs = self.model.get_val_seg_masks(val_imgs)

        # calculate metrics for each model output
        ue_metrics = {}
        # for seg, uncertainty_map in zip(segs_K, uncertainty_maps):
        for output_name, output in outputs.items():
            ue_metrics[output_name] = calculate_ue_metrics(
                                segmentations=output["segs"],
                                labels=val_labels,
                                uncertainty_maps=output["uncertainty_maps"],
                                max_uncertainty=opt.max_uncertainty,
                                num_thresholds=opt.num_thresholds,
                                threshold_type=opt.threshold_type,
                                )
            ue_metrics[output_name]["miou"] = calculate_miou(segmentations=output["segs"], labels=val_labels, num_classes=len(self.known_class_list)+1)
        return ue_metrics

    def init_val_ue_metrics(self, num_thresholds):
        metric_names = ["n_inaccurate_and_certain", "n_accurate_and_certain", "n_uncertain_and_accurate", "n_uncertain_and_inaccurate", "miou"]

        # if metrics_totals[output_name], and output_name is not in metrics_totals, then it will return a dict with the keys being the metric names, and the values being zeros
        metrics_totals = defaultdict(lambda: {metric_name: torch.zeros(num_thresholds) for metric_name in metric_names})
        metrics_counts = defaultdict(lambda: {metric_name: 0 for metric_name in metric_names})

        return metrics_totals, metrics_counts

    @torch.no_grad()
    def validate_uncertainty_estimation(self, val_dataset, test_count=0):
        print(f"\nValidating uncertainty estimation on {val_dataset.name}")

        # init running totals
        val_metrics_totals, val_metrics_counts = self.init_val_ue_metrics(self.opt.num_thresholds)

        dataloader = self._init_dataloader(val_dataset)
        iterator = tqdm(dataloader)
        for val_dict in iterator:
            val_imgs = to_device(val_dict["img"], self.device)
            val_labels = to_device(val_dict["label"], self.device)

            # calculate ue metrics from batch
            val_metrics_dict = self.validate_batch(val_imgs, val_labels, self.opt)
            
            # update running totals
            update_running_totals(val_metrics_totals, val_metrics_counts, val_metrics_dict)

        # calculate metrics from raw results
        processed_metrics = {}
        for output_name in val_metrics_totals:
            processed_metrics[output_name] = calculate_metrics_suite(val_metrics_totals[output_name], val_metrics_counts[output_name], self.states)

        # plot metrics to wandb
        plot_ue_metrics(processed_metrics, test_count, dataset_name=val_dataset.name, plot_plots=False)
    ######################################################################################################################################################
        

    def test(self, test_count=0):
        # validate uncertainty estimation
        for val_dataset in self.val_datasets:
            self.validate_uncertainty_estimation(val_dataset, test_count=test_count)
