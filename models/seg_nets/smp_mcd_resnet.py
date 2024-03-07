import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torchvision.models.resnet import ResNet

class MCDResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, dropout_prob, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        
        self.dropout_prob = dropout_prob

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            if i > 2:       # not applying dropout to first three layers (including identify mapping)
                x = F.dropout2d(stages[i](x), self.dropout_prob, inplace=True)
            else:
                x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)