import torch
import torch.nn as nn



class BaseModel():
    def __init__(self, *args, **kwargs):

        self.patch_size = None
                
        self._init_device()
        self._init_seg_net()

    def _init_device(self):
        # if available (and not overwridden by opt.use_cpu) use GPU, else use CPU
        if torch.cuda.is_available() and self.opt.use_cpu == False:
            device_id = "cuda:" + self.opt.gpu_no
        else:
            device_id = "cpu"
        
        self.device = torch.device(device_id)

    def _init_seg_net(self):
        self.seg_net = None
        raise NotImplementedError
    
    def model_to_train(self):
        self.seg_net.train()

    def model_to_eval(self):
        self.seg_net.eval()

    def get_seg_masks(self, imgs):
        raise NotImplementedError
    
    def get_val_seg_masks(self, imgs):
        raise NotImplementedError


if __name__ == "__main__":
    
    model = BaseModel(device="cpu")