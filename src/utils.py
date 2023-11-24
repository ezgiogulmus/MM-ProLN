import os
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import torch

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, recall_score, roc_auc_score, precision_score

from torcheval.metrics import BinaryAccuracy, BinaryAUROC
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    scores['prec'] = precision_score(y_true, y_pred)
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
    # Create directory for the run
    if config['run_name'] is None:
        config['run_name'] = 'run_%s' % (time.time())
    
    config["model_dir"] = os.path.join("./results/", config["run_name"])
    os.makedirs(config["model_dir"], exist_ok=True)

    # Save config dict
    with open(os.path.join(config["model_dir"], "config.json"), 'w') as f:
        json.dump(config, f)
    
    return config

def loop(config, loader, model, criterion, optimizer=None, training=True, return_preds=False):
    avg_meter = AverageMeter()

    all_targets, all_outputs = [], []
    model.train(training)
    for inputs, target, ids in tqdm(loader):
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
    log = compute_all_scores([all_targets, all_outputs])
    log["loss"] = avg_meter.avg
    if return_preds:
        return log, [all_targets, all_outputs]
    return log
