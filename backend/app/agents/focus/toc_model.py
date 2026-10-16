"""FOCUS foot predictor network (TOC + surface normals + foot mask).

Adapted from FOCUS (github.com/OllieBoyne/FOCUS, MIT licence, Boyne & Cipolla, 3DV 2025) and the
DenseDepth decoder of DSINE (github.com/baegwangbin/DSINE). The DSINE code is licensed by Imperial
College for non-commercial, internal or academic research use only; commercial use requires a licence
from researchcontracts.engineering@imperial.ac.uk.
"""

from __future__ import annotations

from pathlib import Path

import geffnet
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS = {5: [2048, 176, 64, 40, 24]}


class Encoder(nn.Module):
    def __init__(self, B: int = 5):
        super().__init__()
        base = geffnet.create_model(f"tf_efficientnet_b{B}_ap", pretrained=False)
        base.global_pool = nn.Identity()
        base.classifier = nn.Identity()
        self.original_model = base

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for vi in v._modules.values():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UpSampleBN(nn.Module):
    def __init__(self, skip_input: int, output_features: int):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode="bilinear", align_corners=True)
        return self._net(torch.cat([up_x, concat_with], dim=1))


def _head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, output_dim, 1),
    )


def _activation(out):
    normal = F.normalize(out[:, :3], p=2, dim=1)
    kappa = F.elu(out[:, [3]]) + 1.0
    toc = torch.sigmoid(out[:, 4:7])
    toc_unc = torch.tanh(out[:, 7:10]) * 10
    hm = torch.sigmoid(out[:, [10]])
    return torch.cat([normal, kappa, toc, toc_unc, hm], dim=1)


class Decoder(nn.Module):
    def __init__(self, num_classes: int, B: int = 5, NF: int = 2048, down: int = 2, learned_upsampling: bool = False):
        super().__init__()
        ch = INPUT_CHANNELS[B]
        self.conv2 = nn.Conv2d(ch[0], NF, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(NF + ch[1], NF // 2)
        self.up2 = UpSampleBN(NF // 2 + ch[2], NF // 4)
        self.up3 = UpSampleBN(NF // 4 + ch[3], NF // 8)
        self.up4 = UpSampleBN(NF // 8 + ch[4], NF // 16)
        self.downsample_ratio = down
        self.pred_head = _head(NF // 16, 128, num_classes)
        self.learned_upsampling = learned_upsampling
        if learned_upsampling:
            self.mask_head = _head(NF // 16, 128, 9 * down * down)

    def forward(self, features):
        b0, b1, b2, b3, b4 = features[4], features[5], features[6], features[8], features[11]
        x = self.conv2(b4)
        x = self.up1(x, b3)
        x = self.up2(x, b2)
        x = self.up3(x, b1)
        x = self.up4(x, b0)
        out = _activation(self.pred_head(x))
        if not self.learned_upsampling:
            return F.interpolate(out, scale_factor=self.downsample_ratio, mode="bilinear", align_corners=False)
        k = self.downsample_ratio
        B, C, H, W = out.shape
        mask = torch.softmax(self.mask_head(x).view(B, 1, 9, k, k, H, W), dim=2)
        up = F.unfold(out, [3, 3], padding=1).view(B, C, 9, 1, 1, H, W)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(B, C, k * H, k * W)


class DenseDepth(nn.Module):
    def __init__(self, num_classes: int, B: int, NF: int, down: int, learned_upsampling: bool):
        super().__init__()
        self.encoder = Encoder(B)
        self.decoder = Decoder(num_classes, B, NF, down, learned_upsampling)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FootPredictorModel(nn.Module):
    OUTPUT_DIM = 11  # normal(3) + kappa(1) + TOC(3) + TOC log-variance(3) + mask(1)

    def __init__(self, arch: str = "densedepth__B5__NF2048__down2__bilinear__BN", **_):
        super().__init__()
        _, B, NF, down, learned, _bn = arch.split("__")
        self.n_net = DenseDepth(self.OUTPUT_DIM, int(B[1:]), int(NF[2:]), int(down[4:]), learned == "learned")
        pool = 16
        self.footedness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool, pool)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.OUTPUT_DIM * pool * pool, 2),
            nn.Sigmoid(),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor) -> dict:
        """img: [B, 3, H, W] RGB in 0..1."""
        res = self.n_net((img - self.mean) / self.std)
        return {
            "normal": res[:, :3],
            "kappa": res[:, [3]],
            "toc": res[:, 4:7],
            "toc_log_var": res[:, 7:10],
            "mask": res[:, [10]],
            "footedness": self.footedness_head(res),
        }

    @classmethod
    def load(cls, path: Path) -> "FootPredictorModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**data["params"])
        model.load_state_dict(data["state_dict"], strict=True)
        return model.eval()
