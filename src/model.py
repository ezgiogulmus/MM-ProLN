import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import models


class FeatExZ(nn.Module):
    def __init__(self, config, channels):
        super(FeatExZ, self).__init__()
        if config["activation"] == "relu":
            self.act_layer = nn.ReLU()
        elif config["activation"] == "leaky_relu":
            self.act_layer = nn.LeakyReLU()
        elif config["activation"] == "gelu":
            self.act_layer = nn.GELU()
        

        str_size = np.clip(math.floor(math.log2(200/config["fe1_depth"])), a_max=6, a_min=2)
        layers = []
        for i in range(config["fe1_depth"]):
            layers.append(nn.Conv3d(channels, channels, kernel_size=3, stride=(str_size, 1, 1), padding=1))
            layers.append(nn.BatchNorm3d(channels))
            layers.append(self.act_layer)
        self.from3d = nn.Sequential(*layers)
        self.avg_pooling = nn.AdaptiveAvgPool3d((1, config["image_size"], config["image_size"]))
        self.apply(init_weights_he_normal)

    def forward(self, x):
        x = self.from3d(x)
        x = self.avg_pooling(x)
        return torch.squeeze(x, 2)


class ImageFusion(nn.Module):
    def __init__(self, mode="concat"):
        super(ImageFusion, self).__init__()
        self.mode = mode
        if mode == "pet_weighted":
            self.weights = [0.8, 0.2]
            self.mode = "average"
        elif mode == "ct_weighted":
            self.weights = [0.2, 0.8]
            self.mode = "average"
        else:
            self.weights = [0.5, 0.5]
        if self.mode == "adaptive":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, pet, ct):
        if self.mode == "concat":
            return torch.cat([pet, ct], 1)
        elif self.mode == "average":
            return sum([w * img for w, img in zip(self.weights, [pet, ct])])
        elif self.mode == "adaptive":
            return self.alpha * pet + self.beta * ct
        elif self.mode == "multiply":
            return pet * ct
        else:
            raise NotImplementedError("Fusion mode not implemented!")


class MMFusion(nn.Module):
    def __init__(self, config, mode, channel, image_size):
        super(MMFusion, self).__init__()
        self.mode = mode
        self.img_dims = (-1, channel, image_size, image_size)
        self.fc_cli1 = nn.Linear(config["cli_size"], int(image_size**2 * channel))

        if self.mode == "adaptive":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, img, cli):
        if self.mode == "summation":
            cli_transformed = self.fc_cli1(cli).view(self.img_dims)
            return img + cli_transformed

        elif self.mode == "concat":
            cli_transformed = self.fc_cli1(cli).view(self.img_dims)
            # print("CLI: ", cli_transformed.shape, img.shape)
            return torch.cat([img, cli_transformed], dim=1)

        elif self.mode == "adaptive":
            cli_transformed = self.fc_cli1(cli).view(self.img_dims)
            return self.alpha * img + self.beta * cli_transformed

        elif self.mode == "multiply":
            cli_transformed = self.fc_cli1(cli).view(self.img_dims)
            return img * cli_transformed
        
        else:
            raise NotImplementedError("Fusion mode not implemented!")


class SmallNetwork(nn.Module):
    def __init__(self, act_layer, z_dim):
        super(SmallNetwork, self).__init__()
        if act_layer == "relu":
            self.act_layer = nn.ReLU()
        elif act_layer == "leaky_relu":
            self.act_layer = nn.LeakyReLU()
        elif act_layer == "gelu":
            self.act_layer = nn.GELU()
        
        self.net = nn.Sequential(
            nn.Conv2d(z_dim, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            self.act_layer,
            nn.Dropout2d(.5),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 256, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2), bias=True),
            nn.BatchNorm2d(256),
            self.act_layer,
            nn.Dropout2d(.5)
        )
        self.apply(init_weights_he_normal)
    def forward(self, x):
        return self.net(x)

def init_backbone(name, pretrained, z_dim, act_layer):
    feat_extractor = None
    last_chn = None
    if name == "small":
        feat_extractor = SmallNetwork(act_layer, z_dim)
        last_chn = 256

    elif name == "resnet":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights)
        backbone.conv1 = nn.Conv2d(z_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feat_extractor = nn.Sequential(*list(backbone.children())[:-2])
        last_chn = 2048

    elif name == "densenet":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights)
        backbone.features.conv0 = nn.Conv2d(z_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feat_extractor = backbone.features
        last_chn = 1024

    elif name == "mobilenet":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights)
        backbone.features[0][0] = nn.Conv2d(z_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        feat_extractor = backbone.features
        last_chn = 576

    elif name == "efficientnet":
        weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_v2_m(weights)
        backbone.features[0][0] = nn.Conv2d(z_dim, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        feat_extractor = backbone.features
        last_chn = 1280

    else:
        raise ValueError(f"Unsupported backbone name: {name}")

    return feat_extractor, last_chn

def init_weights_he_normal(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


class PETModel(nn.Module):
    def __init__(self, config):
        super(PETModel, self).__init__()
        self.config = config

        self.pet = True if "pet" in config["image_data"] else False
        self.ct = True if "ct" in config["image_data"] else False
        self.cli = True if config["cli_size"] > 0 else False

        self.fusion = config["image_fusion"] if config["image_data"] == "pet+ct" else "no_fusion"
        if self.fusion != "no_fusion":
            self.fuser = ImageFusion(config["fusion_method"])

        chns = 2 if self.fusion == "early" and config["fusion_method"] == "concat" else 1
        self.slice_feats = FeatExZ(config, chns)
        chns = 2 if self.fusion in ["early", "mid"] and config["fusion_method"] == "concat" else 1

        if config["cli_size"] > 0:
            self.mm_fuser1 = MMFusion(config, config["mm_fusion"], chns, config["image_size"])
            if config["mm_fusion"] == "concat":
                chns *= 2        
        self.spatial_feats, last_chn = init_backbone(config["backbone"], config["pretrained"], chns, config["activation"])

        last_chn = last_chn*2 if self.fusion == "late" and config["fusion_method"] == "concat" else last_chn
        
        if config["cli_size"] > 0:
            self.mm_fuser2 = MMFusion(config, config["mm_fusion"], last_chn, int(config["image_size"]/32))
            if config["mm_fusion"] == "concat":
                last_chn *= 2
        self.fc = nn.Linear(last_chn, 1, bias=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.sigmoid = True if config["class_weight"] is None else False
        
        if not config["pretrained"]:
            self.apply(init_weights_he_normal)
        
    def forward(self, inputs, return_feats=False):
        pet = inputs[0]
        ct = inputs[1]
        if self.fusion == "early":
            data = [self.fuser(pet=pet, ct=ct)]
        else:
            data = []
            if self.pet:
                data.append(pet)
            if self.ct:
                data.append(ct)

        data2d = [self.slice_feats(d) for d in data]
        
        if self.cli:
            cli = inputs[-1]
            if self.fusion == "mid":
                data2d = [self.fuser(pet=data2d[0], ct=data2d[1])]
            data2d = [self.mm_fuser1(cli=cli, img=d) for d in data2d]
        
        data1d = [self.spatial_feats(d) for d in data2d]
        if self.config["backbone"] == "swin":
            data1d = [d.permute(0, 3, 1, 2) for d in data1d]
        
        if self.cli:
            cli = inputs[-1]
            if self.fusion == "late":
                data1d = [self.fuser(pet=data1d[0], ct=data1d[1])]
            data1d = [self.mm_fuser2(cli=cli, img=d) for d in data1d]
            
        assert len(data1d) == 1
        if self.avg_pooling is None:
            out = torch.mean(data1d[0], -1)
        else:
            out = self.flat(self.avg_pooling(data1d[0]))
        
        flat_out = self.fc(out)
        
        if self.sigmoid:
            flat_out = torch.sigmoid(flat_out)
        
        if return_feats:
            return flat_out, out
        return flat_out

