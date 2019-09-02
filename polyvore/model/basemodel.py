"""Basic model utils."""
import logging

import torch
from torch import nn

import polyvore.config as cfg
from polyvore import debugger

NUM_ENCODER = cfg.NumCate
LOGGER = logging.getLogger(__name__)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight is not None:
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


class LatentCode(nn.Module):
    """Basic class for learning latent code."""

    def __init__(self, param):
        """Latent code.

        Parameters:
        -----------
        See option.param.NetParam
        """
        super().__init__()
        self.param = param
        self.register_buffer("scale", torch.ones(1))

    def debug(self):
        raise NotImplementedError

    def set_scale(self, value):
        """Set the scale of TanH layer."""
        self.scale.fill_(value)

    def feat(self, x):
        """Compute the feature of all images."""
        raise NotImplementedError

    def forward(self, x):
        """Forward a feature from DeepContent."""
        x = self.feat(x)
        if self.param.without_binary:
            return x
        if self.param.scale_tanh:
            x = torch.mul(x, self.scale)
        if self.param.binary01:
            return 0.5 * (torch.tanh(x) + 1)
        # shape N x D
        return torch.tanh(x).view(-1, self.param.dim)


class TxtEncoder(LatentCode):
    def __init__(self, in_feature, param):
        super().__init__(param)
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, param.dim, bias=False),
        )

    def debug(self):
        debugger.log("item.s")

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[0].weight.data, std=0.01)
        nn.init.constant_(self.encoder[0].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class ImgEncoder(LatentCode):
    """Module for encoder to learn the latent codes."""

    def __init__(self, in_feature, param):
        """Initialize an encoder.

        Parameter
        ---------
        in_feature: feature dimension for image features
        param: see option.param.NetParam for details

        """
        super().__init__(param)
        half = in_feature // 2
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_feature, half),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(half, param.dim, bias=False),
        )

    def debug(self):
        debugger.log("item.v")

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[1].weight.data, std=0.01)
        nn.init.constant_(self.encoder[1].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class UserEncoder(LatentCode):
    """User embedding layer."""

    def __init__(self, param):
        """User embedding.

        Parameters:
        ----------
        param: see option.NetParam for details

        """
        super().__init__(param)
        if param.share_user:
            self.encoder = nn.Sequential(
                nn.Linear(param.num_users, 128),
                nn.Softmax(dim=1),
                nn.Linear(128, param.dim, bias=False),
            )
        else:
            self.encoder = nn.Linear(param.num_users, param.dim, bias=False)

    def debug(self):
        debugger.log("user")

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for user encoder."""
        if self.param.share_user:
            nn.init.normal_(self.encoder[0].weight.data, std=0.01)
            nn.init.constant_(self.encoder[0].bias.data, 0.0)
            nn.init.normal_(self.encoder[-1].weight.data, std=0.01)
        else:
            nn.init.normal_(self.encoder.weight.data, std=0.01)


class CoreMat(nn.Module):
    """Weighted hamming similarity."""

    def __init__(self, dim):
        """Weights for this layer that is drawn from N(mu, std)."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.init_weights()

    def debug(self):
        weight = self.weight.data.view(-1).tolist()
        msg, args = debugger.code_to_str(weight)
        LOGGER.debug("Core Mat:" + msg, *args)

    def init_weights(self):
        """Initialize weights."""
        self.weight.data.fill_(1.0)

    def forward(self, x):
        """Forward."""
        return torch.mul(x, self.weight)

    def __repr__(self):
        """Format string for module CoreMat."""
        return self.__class__.__name__ + "(dim=" + str(self.dim) + ")"


class LearnableScale(nn.Module):
    def __init__(self, init=1.0):
        super(LearnableScale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(init))

    def debug(self):
        LOGGER.debug("Core Mat: %.3f", self.weight.item())

    def forward(self, inputs):
        return self.weight * inputs

    def init_weights(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
