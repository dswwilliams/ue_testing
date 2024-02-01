import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from collections import defaultdict
from ue_testing.test_utils import calculate_ue_metrics, calculate_miou, calculate_metrics_suite, plot_ue_metrics, update_running_totals
from ue_testing.test_utils import init_val_ue_metrics, validate_batch
from ue_testing.device_utils import to_device
from datasets.val_datasets import ValDataset
from colourisation_utils import denormalise, norm_by_ranking, quantize_by_ranking

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
            if wandb.run is None:
                # only initialise if we don't have a wandb run already
                wandb.init(project=self.opt.wandb_project, config=self.opt)
            self.vis = None
        else:
            import visdom
            self.vis = visdom.Visdom(port=8097)
            envs = set(self.vis.get_env_list())
            envs.remove("main")
            for env in envs:
                self.vis.delete_env(env)
            self.vis.close()    # clear windows


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
            if self.vis:
                self.vis.fork_env("main", dataset.name)

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
    def validate_uncertainty_estimation(self, val_dataset, test_count=0):
        print(f"\nValidating uncertainty estimation on {val_dataset.name}")

        # init running totals
        val_metrics_totals, val_metrics_counts = init_val_ue_metrics(self.opt.num_thresholds)

        dataloader = self._init_dataloader(val_dataset)
        iterator = tqdm(dataloader)
        for step, val_dict in enumerate(iterator):
            val_imgs = to_device(val_dict["img"], self.device)
            val_labels = to_device(val_dict["label"], self.device)

            # calculate ue metrics from batch
            val_metrics_dict = validate_batch(val_imgs, val_labels, self.model, self.opt)
            
            # update running totals
            update_running_totals(val_metrics_totals, val_metrics_counts, val_metrics_dict)

            if step >= 3:
                break

        # calculate metrics from raw results
        processed_metrics = {}
        for output_name in val_metrics_totals:
            processed_metrics[output_name] = calculate_metrics_suite(val_metrics_totals[output_name], val_metrics_counts[output_name], self.states)

        # plot metrics to wandb
        plot_ue_metrics(processed_metrics, test_count, dataset_name=val_dataset.name, plot_plots=True, vis=self.vis)
    ######################################################################################################################################################
        
    def get_qual_results(self, test_count=0):
        # validate uncertainty estimation
        for val_dataset in self.val_datasets:
            self.view_qual_results(val_dataset, test_count=test_count)

    @torch.no_grad()
    # def view_val_segmentations(self, val_dataset, model, training_it_count, masking_model=None):
    def view_qual_results(self, val_dataset, test_count=0):
        """
        - extract and view qualitative segmentation and uncertainty estimation results
        """
        dataset_name = val_dataset.name

        print("viewing val segmentations for {}".format(dataset_name))


        # creating dataloader
        self.val_seg_idxs[dataset_name] = [int(idx) for idx in self.val_seg_idxs[dataset_name]]
        val_dataset = torch.utils.data.Subset(val_dataset, self.val_seg_idxs[dataset_name])
        _num_workers = 0 if self.opt.num_workers == 0 else 2
        qual_dataloader = torch.utils.data.DataLoader(
                                                    dataset=val_dataset, 
                                                    batch_size=min(self.opt.batch_size, len(val_dataset)), 
                                                    shuffle=False, 
                                                    num_workers=_num_workers, 
                                                    drop_last=False,
                                                    )
        iterator = tqdm(qual_dataloader)

        seg_count = 0
        for _, (val_dict) in enumerate(iterator):
            val_imgs = to_device(val_dict["img"], self.device)
            val_labels = to_device(val_dict["label"], self.device)

            """
            - what are the relevant seg_masks to look at while validating?
            - i think in the gssl cases, we want to look at query and target seg masks
            - but if its just a standard segmentation model, we just want to look at "vanilla" seg masks
            - this is defined in model and get_val_seg_masks
            """

            seg_outputs = self.model.get_val_seg_masks(val_imgs)

            # reverse image normalisation, ready for viewing
            val_imgs = denormalise(val_imgs, self.opt.use_imagenet_norm).permute(0,2,3,1).detach().cpu().numpy()    # [bs, h, w, 3]
            for batch_no in range(val_imgs.shape[0]):
                # one wandb log per batch item
                masks_log = {}

                val_img = val_imgs[batch_no]        # [h, w, 3]
                ground_truth_mask = val_labels[batch_no].detach().cpu().numpy()
                masks_log["ground_truth"] = {"mask_data": ground_truth_mask, "class_labels": self.class_dict}

                for output_name in seg_outputs:
                    segs = seg_outputs[output_name]["segs"].detach().cpu().numpy()
                    seg = segs[batch_no]
                    uncertainty_maps = seg_outputs[output_name]["uncertainty_maps"]

                    confidences = quantize_by_ranking(1 - uncertainty_maps, n_bins=10).detach().cpu().numpy()
                    confidence = confidences[batch_no]


                    masks_log[f"{output_name}_seg"] = {"mask_data": seg, "class_labels": self.class_dict}
                    masks_log[f"{output_name}_conf"] = {"mask_data": confidence, "class_labels": {idx : str(idx) for idx in range(10)}}
            if self.vis:
                pass
            else:
                masked_image = wandb.Image(
                                    val_img,
                                    masks=masks_log,
                                    )
                wandb.log({f"val_segs {dataset_name}/{seg_count}": masked_image}, commit=False)

                seg_count += 1
    ######################################################################################################################################################
        

