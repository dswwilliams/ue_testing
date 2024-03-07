import sys
import torch
import argparse
import socket
sys.path.append("../")
from tester import Tester
from ue_testing.utils.test_utils import str2bool
torch.set_printoptions(sci_mode=False, linewidth=200)


""" Parsing command line arguments """

parser = argparse.ArgumentParser()
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
parser.add_argument('--model_arch', type=str, default="vit_m2f", help="model architecture: vit_m2f or deeplab")
parser.add_argument('--temperature', type=float, default=0.07, help="temperature for output softmax")
parser.add_argument('--sharpen_temp', type=float, default=None, help="temperature for sharpening of output distribution")
parser.add_argument('--nheads', type=int, default=4, help="number of attention heads for ViT encoder")
parser.add_argument('--vit_type', type=str, default="small", help="type of ViT encoder: small, small_mcd, base, base_mcd")
parser.add_argument('--lora_rank', type=int, default=None, help="if not None, use lora with this rank")
parser.add_argument('--prototype_len', type=int, default=256, help="length of prototype features")
parser.add_argument('--intermediate_dim', type=int, default=256, help="length of decoded features")
parser.add_argument('--use_imagenet_norm', type=str2bool, default=True, help="whether to use imagenet mean and std for normalisation")
parser.add_argument('--gamma_scaling', type=str, default="softmax", help="determines whether gamma is calculated for logits or softmax scores")
parser.add_argument('--gamma_temp', type=float, default=0.1, help="if gamma_scaling is softmax, then this is the temperature used")
parser.add_argument('--use_deep_features', type=str2bool, default=True, help="where to extract features from")
parser.add_argument('--use_proto_seg', type=str2bool, default=False, help="whether to use prototype segmentation")
parser.add_argument('--mlp_dropout_prob', type=float, default=0, help="Dropout probability for MLPs in MCD ViT Encoder")
parser.add_argument('--attn_dropout_prob', type=float, default=0.2, help="Dropout probability for attention in MCD ViT Encoder")

# ======================== Validation Options ========================
parser.add_argument('--output_rank_metrics', type=str2bool, default=False, help="normalise uncertainty metric by rank")
parser.add_argument('--val_transforms', type=str2bool, default=False, help="whether to colour-transform val images")
parser.add_argument('--model_type', type=str, default="fake")
parser.add_argument('--val_batch_size', type=int, default=None)
parser.add_argument('--val_every', type=int, default=500, help="frequency of validation w.r.t. number of training iterations")
parser.add_argument('--skip_validation', type=str2bool, default=False, help="whether to skip validation during training")
parser.add_argument('--n_train_segs', type=int, default=4, help="number of qualitative validations viewed from training dataset")
parser.add_argument('--n_val_segs', type=int, default=4, help="number of qualitative validations viewed from val dataset")
parser.add_argument("--max_uncertainty", type=float, default=None, help="upperbound for uncertainty thresholds")
parser.add_argument("--threshold_type", type=str, default="linear", help="how thresholds are distributed: linear or log")
parser.add_argument("--num_thresholds", type=int, default=500, help="number of thresholds used in validation")
parser.add_argument("--uncertainty_metric", type=str, default="max_softmax", help="metric used to calculate uncertainty")
parser.add_argument("--n_mcd_samples", type=int, default=8, help="number of Monte Carlo Dropout samples used")

# ======================== Logging, Loading and Saving ========================
parser.add_argument('--use_wandb', type=str2bool, default=True, help="whether to use wandb for logging, else use visdom")
parser.add_argument('--wandb_project', type=str, default="test", help="name of wandb project")
parser.add_argument('--dino_path', type=str, default="/Users/dw/code/pytorch/gammassl/models/dinov2.pth", help="path to dino model weights")
parser.add_argument('--dino_repo_path', type=str, default="../dinov2", help="path to dino repo")
parser.add_argument('--save_path', type=str,  default=None, help="path from which to load saved model")
parser.add_argument('--prototypes_path', type=str,  default=None, help="path from which to load saved prototypes")





if __name__ == "__main__":
    # parse arguments
    opt = parser.parse_args()
    
    # import and init model
    if opt.model_type == "gammassl":
        from models.gammassl_model import GammaSSLModel
        model = GammaSSLModel(opt)
    elif opt.model_type == "mcd":
        from models.mcd_model import MCDModel
        model = MCDModel(opt)
    elif opt.model_type == "ensemble":
        from models.ensemble_model import EnsembleModel
        model = EnsembleModel(opt)

    # test model according to options in opt
    tester = Tester(opt, model)
    tester.test()