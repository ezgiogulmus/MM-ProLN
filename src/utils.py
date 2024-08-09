import os
import numpy as np
import random
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from src.datagen import ProstateImages
from src.model import PETModel, MLP
    
def get_model(config):
    if config["image_data"] in ["pet", "ct", "pet+ct"]:
        model = PETModel(config)
    else:
        model = MLP(config)
    
    model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    if config["load_from"] and os.path.isfile(os.path.join(config["load_from"])):
        model.load_state_dict(torch.load(os.path.join(config["load_from"])))
        print("Loaded model from: ", os.path.join(config["load_from"]))
        
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

def compute_all_scores(preds):
    """
    Computes all the scores for the given predictions and labels
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: a dictionary with all the scores
    """
    
    y_true, y_pred = preds
    scores = {}
    scores['auc'] = roc_auc_score(y_true, y_pred)
    
    y_pred = np.array([y>.5 for y in y_pred])
    y_true = np.array(y_true)
    
    scores['acc'] = accuracy_score(y_true, y_pred)
    scores['prec'] = precision_score(y_true, y_pred, zero_division=0)
    scores['recall'] = recall_score(y_true, y_pred)
    scores['f1'] = f1_score(y_true, y_pred)
    scores['mcc'] = matthews_corrcoef(y_true, y_pred)
    

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    scores["spec"] = tn / (tn + fp)
    scores["sens"] = tp / (tp + fn)
    return scores

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def create_run(config):
    run_list = [i for i in [config["run_name"], config["backbone"], config["image_data"], config["image_fusion"], config["fusion_method"], config["clinical_group"], config["mm_fusion"]] if i is not None]
    config["run_name"] = ("_").join(run_list)
    
    config["model_dir"] = os.path.join(config["results_folder"], config["run_name"])
    os.makedirs(config["model_dir"], exist_ok=True)

    # Save config dict
    with open(os.path.join(config["model_dir"], "config.json"), 'w') as f:
        json.dump(config, f)
    
    return config

def get_dataloaders(config, train_df, train_idx, val_idx, test_df):
    train_data = ProstateImages(config, train_df, "train", train_idx)
    val_data = ProstateImages(config, train_df, "val", val_idx)
    test_data = ProstateImages(config, test_df, "test")
    train_loader = DataLoader(train_data, config["batch_size"], num_workers=8, pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, config["val_batch_size"], num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, config["val_batch_size"], num_workers=8, pin_memory=True)
    print("Train steps: ", len(train_loader), "Val steps: ", len(val_loader), "Test steps:", len(test_loader))
    return train_loader, val_loader, test_loader

def loop(config, loader, model, criterion, optimizer=None, training=True, return_preds=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    avg_meter = AverageMeter()

    all_ids, all_targets, all_outputs = [], [], []
    model.train(training)
    for batch_idx, (inputs, target, ids) in enumerate(loader):
        inputs = [i.to(device) for i in inputs]
        [all_targets.append(t.item()) for t in target]
        target = target.to(device)

        with torch.set_grad_enabled(training):
            output = model(inputs)
            loss = criterion(output, target)

        if training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_meter.update(loss.item(), inputs[0].size(0))
        if config["class_weight"] is not None:
            output = torch.sigmoid(output)
        [all_outputs.append(o.item()) for o in output.detach().cpu()]
        all_ids.extend(ids)
        if batch_idx % 5 == 0:
            print(f"\tBatch [{batch_idx}/{len(loader)}] Loss: {avg_meter.avg:.2f}")
    log = compute_all_scores([all_targets, all_outputs])
    log["loss"] = avg_meter.avg
    if return_preds:
        return log, [all_ids, all_targets, all_outputs]
    return log


def set_seed(seed=58):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True