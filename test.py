import sys
import torch
import argparse
import os
import socket
sys.path.append("../")
from tester import Tester

torch.set_printoptions(sci_mode=False, linewidth=200)

########################################################################
## PARSING COMMAND LINE ARGUMENTS ##
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser()
### ###

# ======================== Data Loader and Device  ========================
parser.add_argument('--use_cpu', type=str2bool, default=False, help="override and use cpu")
parser.add_argument('--gpu_no', type=str, default="0", help="which gpu to use")
parser.add_argument('--num_workers', type=int, default=3, help="number of workers for dataloader")

# ======================== Dataset Paths ========================
parser.add_argument('--cityscapes_dataroot', type=str, default="/Volumes/mrgdatastore6/ThirdPartyData/", help="path to cityscapes training dataset")
parser.add_argument('--unlabelled_dataroot', type=str, default="/Volumes/scratchdata/dw/sax_raw", help="path to unlabelled training dataset")
parser.add_argument('--wilddash_dataroot', type=str, default="/Volumes/scratchdata/dw/wilddash/", help="path to wilddash test dataset")
parser.add_argument('--bdd_val_dataroot', type=str, default="/Users/dw/data/bdd_10k", help="path to bdd100k validation dataset")

# ======================== Model ========================
parser.add_argument('--temperature', type=float, default=0.07, help="temperature for output softmax")
parser.add_argument('--sharpen_temp', type=float, default=None, help="temperature for sharpening of output distribution")
parser.add_argument('--nheads', type=int, default=4, help="number of attention heads for ViT encoder")
parser.add_argument('--vit_size', type=str, default="small", help="size of ViT encoder: small or base")
parser.add_argument('--lora_rank', type=int, default=None, help="if not None, use lora with this rank")
parser.add_argument('--prototype_len', type=int, default=256, help="length of prototype features")
parser.add_argument('--intermediate_dim', type=int, default=256, help="length of decoded features")
parser.add_argument('--use_imagenet_norm', type=str2bool, default=True, help="whether to use imagenet mean and std for normalisation")
parser.add_argument('--gamma_scaling', type=str, default="softmax", help="determines whether gamma is calculated for logits or softmax scores")
parser.add_argument('--gamma_temp', type=float, default=0.1, help="if gamma_scaling is softmax, then this is the temperature used")


# ======================== Validation Options ========================
# TODO
parser.add_argument('--output_rank_metrics', type=str2bool, default=False, help="normalise uncertainty metric by rank")
# TODO
parser.add_argument('--val_transforms', type=str2bool, default=False, help="whether to colour-transform val images")
parser.add_argument('--model_type', type=str, default="fake")
parser.add_argument('--val_batch_size', type=int, default=None)
parser.add_argument('--val_every', type=int, default=500, help="frequency of validation w.r.t. number of training iterations")
parser.add_argument('--skip_validation', type=str2bool, default=False, help="whether to skip validation during training")
parser.add_argument('--n_train_segs', type=int, default=4, help="number of qualitative validations viewed from training dataset")
parser.add_argument('--n_val_segs', type=int, default=4, help="number of qualitative validations viewed from val dataset")
parser.add_argument("--max_uncertainty", type=float, default=1, help="upperbound for uncertainty thresholds")
parser.add_argument("--threshold_type", type=str, default="linear", help="how thresholds are distributed: linear or log")
parser.add_argument("--num_thresholds", type=int, default=500, help="number of thresholds used in validation")

# ======================== Logging, Loading and Saving ========================
parser.add_argument('--use_wandb', type=str2bool, default=True, help="whether to use wandb for logging, else use visdom")
parser.add_argument('--wandb_project', type=str, default="test", help="name of wandb project")


opt = parser.parse_args()
if socket.gethostname() == "smaug":
    opt.cityscapes_dataroot = "/home/dsww/data/"
    opt.unlabelled_dataroot = "/mnt/data/bdd100k"
    opt.dino_path = "/home/dsww/networks/dinov2/dinov2.pth"
    opt.bdd_val_dataroot = "/home/dsww/data/bdd_10k"
elif opt.use_cpu:
    opt.batch_size = 2
    opt.num_workers = 0
    opt.cityscapes_dataroot = "/Users/dw/data/"
    opt.unlabelled_dataroot = "/Users/dw/data/bdd100k"
    opt.wilddash_dataroot = "/Users/dw/data/wilddash"
    opt.sax_raw_dataroot = "/Users/dw/data/sax_raw"
    opt.sax_labelled_dataroot = "/Users/dw/data/sax_labelled"
    opt.sensor_models_path = "/Users/dw/code/lut/sensor-models"
# print(opt)
########################################################################



if __name__ == "__main__":
    
    # import and init model
    if opt.model_type == "fake":
        from models.fake_model import FakeModel
        model = FakeModel(opt)
    elif opt.model_type == "gammassl":
        from models.gammassl_model import GammaSSLModel
        model = GammaSSLModel(opt)

    tester = Tester(opt, model)

    # test model
    tester.test()