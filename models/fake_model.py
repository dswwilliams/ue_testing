import torch
from models.base_model import BaseModel 

class FakeModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._init_seg_net()

    def _init_seg_net(self):
        self.seg_net = None

    def get_seg_masks(self, imgs):
        bs,_,h,w = imgs.shape
        seg_masks = torch.randn(bs, 19, h, w)
        return seg_masks
    
    def get_val_seg_masks(self, imgs):
        bs,_,h,w = imgs.shape
        seg_masks = 10 * torch.randn(bs, 19, h, w)

        seg_masks = torch.softmax(seg_masks, dim=1)
        ms_imgs, segs = torch.max(seg_masks, dim=1)

        return {"fake": {"segs": segs, "uncertainty_maps": ms_imgs}}