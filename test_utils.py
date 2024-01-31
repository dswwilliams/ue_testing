import torch
from device_utils import to_device
import matplotlib.pyplot as plt 
import numpy as np


@torch.no_grad()
def calculate_ue_metrics(
                    segmentations, 
                    labels, 
                    num_thresholds=100, 
                    uncertainty_maps=None, 
                    max_uncertainty=1, 
                    threshold_type="linear", 
                    ):

    bs = segmentations.shape[0]
    device = segmentations.device

    ################################################################################################################
    ### defining thresholds ###
    if threshold_type == "log":
        thresholds = max_uncertainty * torch.logspace(start=-15, end=0, steps=num_thresholds, base=2)        # range: [0, max_uncertainty]
    elif threshold_type == "linear":
        thresholds = max_uncertainty * torch.linspace(0, 1, steps=num_thresholds)                            # range: [0, max_uncertainty]
    ################################################################################################################

    ################################################################################################################
    ### init running variables ###
    val_metrics = {}
    val_metrics["n_uncertain_and_accurate"] = to_device(torch.zeros(bs, num_thresholds), device)
    val_metrics["n_uncertain_and_inaccurate"] = to_device(torch.zeros(bs, num_thresholds), device)
    val_metrics["n_inaccurate_and_certain"] = to_device(torch.zeros(bs, num_thresholds), device)
    val_metrics["n_accurate_and_certain"] = to_device(torch.zeros(bs, num_thresholds), device)
    ################################################################################################################

    accuracy_masks = torch.eq(segmentations, labels).float()        # where segmentations == labels, 1, else 0
    # loop over threshold values
    for threshold_no in range(thresholds.shape[0]):
        ################################################################################################################
        ### getting confidence_masks ###
        threshold = thresholds[threshold_no]
        confidence_masks = torch.le(uncertainty_maps, threshold).float()           # 1 if uncertainty_maps <= threshold, else 0
        ################################################################################################################
        
        ################################################################################################################
        ### calculating uncertainty estimation metrics ###
        n_accurate_and_certain = (accuracy_masks * confidence_masks).sum((1,2))
        n_inaccurate_and_certain = ((1-accuracy_masks) * confidence_masks).sum((1,2))
        n_uncertain_and_accurate = (accuracy_masks * (1-confidence_masks)).sum((1,2))
        n_uncertain_and_inaccurate = ((1-accuracy_masks) * (1-confidence_masks)).sum((1,2))
        
        val_metrics["n_inaccurate_and_certain"][:, threshold_no] = n_inaccurate_and_certain
        val_metrics["n_accurate_and_certain"][:, threshold_no] = n_accurate_and_certain
        val_metrics["n_uncertain_and_accurate"][:, threshold_no] = n_uncertain_and_accurate
        val_metrics["n_uncertain_and_inaccurate"][:, threshold_no] = n_uncertain_and_inaccurate
        ################################################################################################################
    return val_metrics


def calculate_miou(segmentations, labels, num_classes):
    total_iou = 0
    n_active_classes = 0
    for k in range(num_classes):
        class_seg_mask = (segmentations == k).float()
        class_label_mask = (labels == k).float()

        intersection = (class_seg_mask * class_label_mask).sum()
        union = torch.max(class_seg_mask, class_label_mask).sum()

        if not (union == 0):
            iou = intersection/union
            total_iou += iou
            n_active_classes += 1

    if n_active_classes == 0:
        miou = 0
    else:
        miou = total_iou/n_active_classes
    return miou

def calculate_precision_recall(metrics_totals, states):
    tp = metrics_totals[states["tp"]]
    fp = metrics_totals[states["fp"]]
    fn = metrics_totals[states["fn"]]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precision[torch.nonzero(precision.isnan())] = 0
    return precision, recall


def calculate_fbeta_score(metrics_totals, states, beta=1):
    tp = metrics_totals[states["tp"]]
    fp = metrics_totals[states["fp"]]
    fn = metrics_totals[states["fn"]]
    fbeta_score = (1+beta**2)*tp /((1+beta**2)*tp + fp + (beta**2)*fn)
    return fbeta_score

def calculate_accuracy(metrics_totals, states):
    tp = metrics_totals[states["tp"]]
    tn = metrics_totals[states["tn"]]
    fp = metrics_totals[states["fp"]]
    fn = metrics_totals[states["fn"]]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

def calculate_p_certain(metrics_totals, states):
    tp = metrics_totals[states["tp"]]
    tn = metrics_totals[states["tn"]]
    fp = metrics_totals[states["fp"]]
    fn = metrics_totals[states["fn"]]
    p_certain = (tp + fp) / (tp + tn + fp + fn)
    return p_certain

def calculate_p_accurate(metrics_totals, states):
    tp = metrics_totals[states["tp"]]
    tn = metrics_totals[states["tn"]]
    fp = metrics_totals[states["fp"]]
    fn = metrics_totals[states["fn"]]
    p_accurate = (tp + fn) / (tp + tn + fp + fn)
    return p_accurate

def calculate_mean_stats(metrics_totals):
    n_accurate_and_certain = metrics_totals["n_accurate_and_certain"]
    n_uncertain_and_inaccurate = metrics_totals["n_uncertain_and_inaccurate"]
    n_inaccurate_and_certain = metrics_totals["n_inaccurate_and_certain"]
    n_uncertain_and_accurate = metrics_totals["n_uncertain_and_accurate"]
    total_sum = n_accurate_and_certain + n_uncertain_and_inaccurate + n_inaccurate_and_certain + n_uncertain_and_accurate

    mean_stats = {}
    mean_stats["mean_accurate_and_certain"] = n_accurate_and_certain / total_sum
    mean_stats["mean_uncertain_and_inaccurate"] = n_uncertain_and_inaccurate / total_sum
    mean_stats["mean_inaccurate_and_certain"] = n_inaccurate_and_certain / total_sum
    mean_stats["mean_uncertain_and_accurate"] = n_uncertain_and_accurate / total_sum
    return mean_stats



def calculate_metrics_suite(val_metrics_totals, val_metrics_counts, states):
    processed_metrics = {}
    
    processed_metrics["precision"], processed_metrics["recall"] = calculate_precision_recall(
                                                                        val_metrics_totals,
                                                                        states,
                                                                        )

    processed_metrics["fhalf"] = calculate_fbeta_score(val_metrics_totals, states, beta=0.5)
    
    processed_metrics["acc_md"] = calculate_accuracy(val_metrics_totals, states)

    processed_metrics["p_certain"] = calculate_p_certain(val_metrics_totals, states)
    
    processed_metrics["p_accurate"] = calculate_p_accurate(val_metrics_totals, states)
    
    mean_stats = calculate_mean_stats(val_metrics_totals)
    processed_metrics.update(mean_stats)

    processed_metrics["miou"] = val_metrics_totals["miou"] / val_metrics_counts["miou"]

    return processed_metrics

def plot_ue_metrics(processed_metrics, validation_count, dataset_name, plot_plots=False, vis=None):
    """
    - if wandb hasnt been initialised, then use visdom
    """
    if vis is None:
        USE_WANDB = True
        import wandb
    else:
        USE_WANDB = False

    for output_name in processed_metrics:
    
        if plot_plots:
            fig, ax = plt.subplots()
            ax.plot(processed_metrics["p_certain"].cpu().numpy(), processed_metrics["acc_md"].cpu().numpy())
            ax.set_xlabel("p_certain")
            ax.set_ylabel("A_md")
            if USE_WANDB:
                wandb.log({f"{output_name} - A_md {dataset_name}/{validation_count}": fig}, commit=False)
            else:
                vis.line(
                    Y=processed_metrics["acc_md"].cpu().numpy(), 
                    X=processed_metrics["p_certain"].cpu().numpy(), 
                    opts=dict(title=f"{output_name} - A_md {dataset_name}/{validation_count}", 
                    win=f"{output_name} - A_md {dataset_name}/{validation_count}",
                    xlabel='p_certain', ylabel='A_md')
                    )

            fig, ax = plt.subplots() 
            ax.plot(processed_metrics["p_certain"].cpu().numpy(), processed_metrics["fhalf"].cpu().numpy())
            ax.set_xlabel("p_certain")
            ax.set_ylabel("F_0.5")
            if USE_WANDB:
                wandb.log({f"{output_name} - F_0.5 {dataset_name}/{validation_count}": fig}, commit=False)
            else:
                vis.line(
                    Y=processed_metrics["fhalf"].cpu().numpy(), 
                    X=processed_metrics["p_certain"].cpu().numpy(), 
                    opts=dict(title=f"{output_name} - F_0.5 {dataset_name}/{validation_count}", 
                    win=f"{output_name} - F_0.5 {dataset_name}/{validation_count}",
                    xlabel='p_certain', ylabel='F_0.5')
                    )

            fig, ax = plt.subplots()
            ax.plot(processed_metrics["recall"].cpu().numpy(), processed_metrics["precision"].cpu().numpy())
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            if USE_WANDB:
                wandb.log({f"{output_name} - Precision vs Recall {dataset_name}/{validation_count}": fig}, commit=False)
            else:
                vis.line(
                    Y=processed_metrics["precision"].cpu().numpy(), 
                    X=processed_metrics["recall"].cpu().numpy(), 
                    opts=dict(title=f"{output_name} - Precision vs Recall {dataset_name}/{validation_count}", 
                    win=f"{output_name} - Precision vs Recall {dataset_name}/{validation_count}",
                    xlabel='recall', ylabel='precision')
                    )


        # plotting aggregated metrics (i.e. single datum per validation step)
        if USE_WANDB:
            wandb.log({f"{output_name} - {dataset_name}/Max A_md": processed_metrics["acc_md"].max().float().item()}, commit=False)
            wandb.log({f"{output_name} - {dataset_name}/Max F_0.5": processed_metrics["fhalf"].max().float().item()}, commit=False)
            wandb.log({f"{output_name} - {dataset_name}/Segmentation Accuracy": processed_metrics["p_accurate"][0].float().item()}, commit=False)
            wandb.log({f"{output_name} - {dataset_name}/Mean IoU": processed_metrics["miou"][0].float().item()}, commit=False)
        else:
            # add point to existing plot in visdom
            vis.line(
                Y=np.array([processed_metrics["acc_md"].max().float().item()]), 
                X=np.array([validation_count]), 
                win=f"{output_name} - {dataset_name}/Max A_md", 
                update='append'
                )
            vis.line(
                Y=np.array([processed_metrics["fhalf"].max().float().item()]), 
                X=np.array([validation_count]), 
                win=f"{output_name} - {dataset_name}/Max F_0.5", 
                update='append'
                )
            vis.line(
                Y=np.array([processed_metrics["p_accurate"][0].float().item()]), 
                X=np.array([validation_count]), 
                win=f"{output_name} - {dataset_name}/Segmentation Accuracy", 
                update='append'
                )
            vis.line(
                Y=np.array([processed_metrics["miou"][0].float().item()]), 
                X=np.array([validation_count]), 
                win=f"{output_name} - {dataset_name}/Mean IoU", 
                update='append'
                )
            

def update_running_totals(val_metrics_totals, val_metrics_counts, val_metrics_dict):
    for output_name, val_metrics in val_metrics_dict.items():
        for metric_name in val_metrics:
            val_metrics_totals[output_name][metric_name] += val_metrics[metric_name].sum(0).cpu()
            if len(val_metrics[metric_name].shape) > 0:
                val_metrics_counts[output_name][metric_name] += val_metrics[metric_name].shape[0]
            else:
                val_metrics_counts[output_name][metric_name] += 1