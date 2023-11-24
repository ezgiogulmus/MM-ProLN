import torch
import torch.nn as nn
import torch.optim as optim
import math

from model import PETModel
from model_mlp import MLP


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model(config):
    if config["image_data"] in ["pet", "ct", "pet+ct"]:
        model = PETModel(config)
    else:
        model = MLP(config)
    if torch.cuda.device_count() > 1:
        print("Running on ", torch.cuda.device_count(), " GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['learning_rate'], weight_decay=config["l2_reg"])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['learning_rate'], weight_decay=config["l2_reg"],
                              nesterov=True, momentum=.9)
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(params, lr=config['learning_rate'], weight_decay=config["l2_reg"])
    else:
        raise NotImplementedError

    
    if config["class_weight"] is None:
        criterion = nn.BCELoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config["class_weight"]]).cuda())
    

    if config['lr_scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-9)
    elif config['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5, threshold_mode="abs", threshold=0.01, verbose=1, min_lr=1e-9)
    elif config['lr_scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    return model, optimizer, criterion, scheduler
