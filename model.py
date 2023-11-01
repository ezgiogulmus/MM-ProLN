from einops import rearrange
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

class PETModel(nn.Module):
    def __init__(self, config):
        super(PETModel, self).__init__()
        self.config = config

        self.pet = True if "pet" in config["image_data"] else False
        self.ct = True if "ct" in config["image_data"] else False
        self.cli = True if config["cli_size"] > 0 else False
        
        self.sigmoid = True if config["class_weight"] is None else False

        self.slice_feats = FeatExZ(config)
        self.backbone, last_chn = init_backbone(config["backbone"], config["pretrained"], config["z_dim"], config["activation"])
        self.fc = nn.Linear(last_chn, 1, bias=True)

        self.fc_cli1 = nn.Linear(config["cli_size"], int(config["image_size"]**2))
        self.fc_cli2 = nn.Linear(config["cli_size"], int((config["image_size"]//32)**2))

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        
        if not config["pretrained"]:
            self.apply(init_weights_he_normal)
        
    def forward(self, inputs, return_feats=False):
        pet = inputs[0]
        ct = inputs[1]
        seg = inputs[2]
        if self.cli:
            cli = inputs[-1]

        outputs = []
        if self.pet:
            pet = self.slice_feats(pet)
        
            if self.cli:
                cli_vector = torch.unsqueeze(self.fc_cli1(cli), 1)
                cli_vector = rearrange(cli_vector, "b c (h w) -> b c h w", h=pet.shape[-1])
                pet = pet + cli_vector
            pet = self.backbone(pet)
            if self.cli:
                cli_vector = torch.unsqueeze(self.fc_cli2(cli), 1)
                cli_vector = rearrange(cli_vector, "b c (h w) -> b c h w", h=pet.shape[-1])
                pet = pet + cli_vector
            out = pet

        if self.ct:
            ct = self.slice_feats(ct)
            if self.cli:
                cli_vector = torch.unsqueeze(self.fc_cli1(cli), 1)
                cli_vector = rearrange(cli_vector, "b c (h w) -> b c h w", h=ct.shape[-1])
                ct = ct + cli_vector
            ct = self.backbone(ct)
            if self.cli:
                cli_vector = torch.unsqueeze(self.fc_cli2(cli), 1)
                cli_vector = rearrange(cli_vector, "b c (h w) -> b c h w", h=ct.shape[-1])
                ct = ct + cli_vector
            out = ct
        
        if self.ct and self.pet:
            out = pet + ct

        flat_out = self.flat(self.avg_pooling(out))
        flat_out = self.fc(flat_out)
        if self.sigmoid:
            flat_out = torch.sigmoid(flat_out)
        if return_feats:
            return flat_out, out
        return flat_out

class FeatExZ(nn.Module):
    def __init__(self, config):
        super(FeatExZ, self).__init__()
        if config["activation"] == "relu":
            self.act_layer = nn.ReLU()
        elif config["activation"] == "leaky_relu":
            self.act_layer = nn.LeakyReLU()
        elif config["activation"] == "gelu":
            self.act_layer = nn.GELU()

        layers = []
        for i in range(config["fe1_depth"]):
            layers.append(nn.Conv3d(1, 1, kernel_size=3, stride=(4, 1, 1), padding=1))
            layers.append(nn.BatchNorm3d(1))
            layers.append(self.act_layer)
        self.from3d = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=7, stride=(4, 1, 1), padding=3),
            nn.BatchNorm3d(1),
            self.act_layer,
            nn.Conv3d(1, 1, kernel_size=5, stride=(4, 1, 1), padding=2),
            nn.BatchNorm3d(1),
            self.act_layer
        )
        self.avg_pooling = nn.AdaptiveAvgPool3d((config["z_dim"], config["image_size"], config["image_size"]))
        self.apply(init_weights_he_normal)

    def forward(self, x):
        x = self.from3d(x)
        x = self.avg_pooling(x)
        return torch.squeeze(x)

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

def init_backbone(name, pretrained, z_dim, act_layer, out_features=1):
    if name == "small":
        feat_extractor = SmallNetwork(act_layer, z_dim)
        last_chn = 256
    elif name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights)
        # backbone.fc = nn.Linear(512, out_features, bias=True)
        backbone.conv1 = nn.Conv2d(z_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feat_extractor = nn.Sequential(*list(backbone.children())[:-2])
        last_chn = 512
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights)
        # backbone.fc = nn.Linear(512, out_features, bias=True)
        backbone.conv1 = nn.Conv2d(z_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feat_extractor = nn.Sequential(*list(backbone.children())[:-2])
        last_chn = 2048
    elif name == "densenet":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights)
        # backbone.classifier = nn.Linear(1024, out_features, bias=True)
        backbone.features.conv0 = nn.Conv2d(z_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        feat_extractor = backbone.features
        last_chn = 1024
    elif name == "mobilenet":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights)
        # backbone.classifier[-1] = nn.Linear(1024, out_features, bias=True)
        backbone.features[0][0] = nn.Conv2d(z_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        feat_extractor = backbone.features
        last_chn = 576
    return feat_extractor, last_chn

def init_weights_he_normal(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
