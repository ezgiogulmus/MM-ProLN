import os
import json
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from datagen import ProstateImages, ProstateData
import networks
import utils
import wandb
pd.options.mode.chained_assignment = None


def main(config=None):
    if config is None:
        initial_config = {
            "data_folder": "./data/",
            "image_folder": "./data/ProstateData/",
            'train_file': "./data/training.csv",
            'test_file': "./data/testing.csv",
            "batch_size": 16,
            "val_batch_size": 8,
            "image_size": 256,
            "early_stopping": 30,
            "cross_validation": -1,
            "lr_scheduler": "ReduceLROnPlateau",
            "epochs": 100,
            "wandb": True,
            "debugging": False,
            "normalization": True,
            "pretrained": False,
            "transform": True,
            "crop": True,
            "load_from": None
        }
        wandb.init(config=initial_config) 
        config = wandb.config
        config["model_dir"] = wandb.run.dir
        print("Model directory is ", config["model_dir"])
        
    else:
        # Create directory for the run
        config['train_file'] = config["data_folder"]+'training.csv'
        config['test_file'] = config["data_folder"]+'testing.csv'
        config = utils.create_run(config)
    print(config["train_file"])
    data = ProstateData(config)
    train_df, test_df = data.selected_feats(config["clinical_group"])
    config = data.config

    config["cli_size"] = train_df.shape[1] - 2
    print("Run config:")
    for k, v in config.items():
        print("\t", k, ":\t", v)

    # Cross-validation
    n_splits = config["cross_validation"] if config["cross_validation"] > 0 else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=58)
    eval_mode = False
    cv_scores = None
    cv_output_auc, cv_output_acc, cv_output = [], [], []
    for cv, (train_idx, val_idx) in enumerate(skf.split(train_df[["idx"]], train_df["nod"])):
        cv += 1
        print("Starting cv fold ", cv)
        model, optimizer, criterion, scheduler = networks.get_model(config)
        if cv == 1:
            print(model)
        if config["load_from"] and os.path.isfile(os.path.join(config["load_from"], f"bestmodel_auc{cv}.pth")):
            model.load_state_dict(torch.load(os.path.join(config["load_from"], f"bestmodel_auc{cv}.pth")))
            print("Loaded model from: ", os.path.join(config["load_from"], f"bestmodel_auc{cv}.pth"))
            eval_mode = True
        # Load clinical data and normalize
        if config["cli_size"] != 0:
            train_df, test_df = data.preprocess(train_df, test_df, train_idx, config)
        # print("Clinical and radiomics: ")
        # for c in train_df.columns:
        #     print(c)
        if config["debugging"]:
            train_idx = list(range(20))
            val_idx = list(range(20, 40))
        train_data = ProstateImages(config, train_df, "train", train_idx)
        val_data = ProstateImages(config, train_df, "val", val_idx)
        test_data = ProstateImages(config, test_df, "test")
        
        train_loader = DataLoader(train_data, config["batch_size"], drop_last=True, shuffle=True)
        val_loader = DataLoader(val_data, config["val_batch_size"], shuffle=True)
        test_loader = DataLoader(test_data, config["val_batch_size"])
        print("Train steps: ", len(train_loader), "Val steps: ", len(val_loader), "Test steps:", len(test_loader))

        log = None
        best_auc, best_acc, trigger = 0, 0, 0
        best_loss = np.inf

        if eval_mode:
            val_log, val_preds = utils.loop(config, val_loader, model, criterion, training=False, return_preds=True, threshold=config["threshold_for_class1"])
            val_scores = utils.compute_all_scores(val_preds)

            print("Validation: ", end="")
            for idx, (k, v) in enumerate(val_scores.items()): 
                ending = "\n" if idx == len(val_scores)-1 else " - "
                print(k, round(v, 4), end=ending)
            
            test_log, test_preds = utils.loop(config, test_loader, model, criterion, training=False, return_preds=True, threshold=config["threshold_for_class1"])
            pd.DataFrame(np.array(test_preds).T, columns=["target", "prediction"]).to_csv(os.path.join(config["model_dir"], f"predictions_{cv}.csv"), index=False)

            test_scores = utils.compute_all_scores(test_preds)
            print("Test results:")
            for idx, (k, v) in enumerate(test_scores.items()): 
                ending = "\n" if idx == len(test_scores)-1 else " - "
                print(k, round(v, 4), end=ending)
            
            if cv_scores is None:
                cv_scores = {"val_"+k: [v] for k, v in val_scores.items()}
                cv_scores.update({"test_"+k: [v] for k, v in test_scores.items()})
            else:
                [cv_scores["val_"+k].append(v) for k, v in val_scores.items()]
                [cv_scores["test_"+k].append(v) for k, v in test_scores.items()]
            pd.DataFrame(cv_scores).to_csv(os.path.join(config["model_dir"], "all_cv_scores.csv"), index=False)

            continue

        for epoch in range(config['epochs']):
            print('Epoch [%d/%d]' % (epoch+1, config['epochs']))

            # train for one epoch
            train_log = utils.loop(config, train_loader, model, criterion, optimizer, threshold=config["threshold_for_class1"])
            print("Train: ", end="")
            for idx, (k, v) in enumerate(train_log.items()):
                ending = "\n" if idx == len(train_log)-1 else " - "
                print(k, round(v, 4), end=ending)

            val_log = utils.loop(config, val_loader, model, criterion, training=False, threshold=config["threshold_for_class1"])
            print("Validation: ", end="")
            for idx, (k, v) in enumerate(val_log.items()): 
                ending = "\n" if idx == len(val_log)-1 else " - "
                print(k, round(v, 4), end=ending)

            if config["wandb"]:    
                wandb.log({"train_"+x: train_log[x] for x in train_log.keys()})
                wandb.log({"val_"+x: val_log[x] for x in val_log.keys()})
            else:
                if log is None:
                    log = {
                        "epoch": [],
                        "lr": []
                    }
                    log.update({"train_"+k: [] for k in train_log.keys()})
                    log.update({"val_"+k: [] for k in val_log.keys()})

                log['epoch'].append(epoch)
                log['lr'].append(optimizer.param_groups[0]['lr'])
                for k, v in log.items():
                    if "train" in k:
                        v.append(train_log[k.replace("train_", "")])
                    elif "val" in k:
                        v.append(val_log[k.replace("val_", "")])

                pd.DataFrame(log).to_csv(os.path.join(config["model_dir"], f"log_{cv}.csv"), index=False)
            
            if config['lr_scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            elif config['lr_scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])

            if epoch > 5:
                trigger += 1
                
                if val_log["auc"] - best_auc > .01 and best_acc - val_log["acc"] < .05:
                    torch.save(model.state_dict(), os.path.join(config["model_dir"], f"bestmodel_auc{cv}.pth"))
                    best_auc = val_log["auc"]
                    
                    best_log_keys = ["best_val_"+k for k, v in val_log.items()]
                    best_log_auc = [v for k, v in val_log.items()]
                    
                    print("=> saved model with the best auc")
                    trigger = 0
                
                if val_log["acc"] - best_acc > .01 and best_auc - val_log["auc"] < .05:
                    torch.save(model.state_dict(), os.path.join(config["model_dir"], f"bestmodel_acc{cv}.pth"))
                    best_acc = val_log["acc"]
                    
                    best_log_keys = ["best_val_"+k for k, v in val_log.items()]
                    best_log_acc = [v for k, v in val_log.items()]
                    
                    print("=> saved model with the best acc")


                elif config['early_stopping'] >= 0:
                    print('early stopping count: ', trigger, '/', config['early_stopping'])

                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

            torch.cuda.empty_cache()
        
        # Best auc model
        model.load_state_dict(torch.load(os.path.join(config["model_dir"], f"bestmodel_auc{cv}.pth")))
        test_log = utils.loop(config, test_loader, model, criterion, training=False, threshold=config["threshold_for_class1"])
        print("Test results:")
        for k, v in test_log.items():
            best_log_auc.append(v)
            best_log_keys.append("test_"+k)
            print("\t", k, ": ", round(v, 4))
        
        if config["wandb"]:
            wandb.log({k: v for k, v in list(zip(best_log_keys, best_log_auc))})

        else:
            cv_output_auc.append(best_log_auc)
            pd.DataFrame(cv_output_auc, columns=best_log_keys).to_csv(os.path.join(config["model_dir"], "cv_output_auc.csv"), index=False)
        
        # Best acc model
        model.load_state_dict(torch.load(os.path.join(config["model_dir"], f"bestmodel_acc{cv}.pth")))
        test_log = utils.loop(config, test_loader, model, criterion, training=False, threshold=config["threshold_for_class1"])
        print("Test results:")
        for k, v in test_log.items():
            best_log_acc.append(v)
            print("\t", k, ": ", round(v, 4))
        
        if config["wandb"]:
            wandb.log({k: v for k, v in list(zip(best_log_keys, best_log_acc))})

        else:
            cv_output_acc.append(best_log_acc)
            pd.DataFrame(cv_output_acc, columns=best_log_keys).to_csv(os.path.join(config["model_dir"], "cv_output_acc.csv"), index=False)
        

        if config["cross_validation"] < 0:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--wandb', default=False, action="store_true")
    parser.add_argument('--debugging', default=False, action="store_true")
    parser.add_argument('--run_config_file', default=None)
    parser.add_argument('--project_name', default=None)
    parser.add_argument('--data_folder', default="./data/")
    parser.add_argument('--image_folder', default="./data/ProstateData/")
    parser.add_argument('--load_from', default=None)

    parser.add_argument('--class_weight', default=None, type=float)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--z_dim', default=32, type=int)
    
    parser.add_argument('--activation', default='leaky_relu', choices=["relu", "leaky_relu", "gelu"])
    parser.add_argument('--backbone', default="small", choices=['small', 'resnet18', "resnet50", "densenet", "mobilenet"])
    parser.add_argument('--pretrained', default=True, action='store_false')
    parser.add_argument('--clinical_group', default="clinical+pet", choices=["None", "clinical", "clinical+radiomics", "clinical+pet", "clinical+ct"])
    parser.add_argument('--image_data', default="pet", choices=["pet", "ct", "pet+ct"])

    parser.add_argument('--transform', default=True, action='store_false')
    parser.add_argument('--crop', default=True, action='store_false')
    parser.add_argument('--normalization', default=True, action='store_false')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--val_batch_size', default=8, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--cross_validation", default=5, type=int)
    
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--learning_rate', default=0.027, type=float)
    parser.add_argument('--lr_scheduler', default="ReduceLROnPlateau", choices=['CosineAnnealingLR', 'ReduceLROnPlateau', "ConstantLR"])
    parser.add_argument('--l2_reg', default=0.027, type=float)
    parser.add_argument('--early_stopping', default=30, type=int)
    parser.add_argument('--threshold_for_class1', default=.5, type=float)
    
    config = parser.parse_args()

    return config


if __name__ == '__main__':
    config = vars(parse_args())
    if config["run_config_file"]:
        new_run_name = config["run_name"]
        load_from = config["load_from"]
        wandb = config["wandb"]
        cross_validation = config["cross_validation"]
        with open(config["run_config_file"], "r") as f:
            config.update(json.load(f))
        config.update({
            "run_name": new_run_name,
            "load_from": load_from,
            "wandb": wandb,
            "cross_validation": cross_validation,
        })
    if config["wandb"]:
        # os.environ["WANDB_API_KEY"] = ## Add your wandb key
        
        parameter_dict = {
            "backbone": {
                "values": ["small", "mobilenet", "resnet18"]
            },
            "z_dim": {
                "values": [8, 16, 32, 64]
            },
            'class_weight': {
                'values': [None, 3, 5]
            },
            "activation": {
                "values": ["leaky_relu", "relu", "gelu"]
            },
            # "pretrained": {
            #     "values": [True, False]
            # },
            # "transform": {
            #     "values": [True, False]
            # },
            # "crop": {
            #     "values": [True, False]
            # },
            "clinical_group": {
                "values": ["clinical", "clinical+radiomics", "clinical+pet", "clinical+ct"],
            },       
            "image_data": {
                "values": ["pet", "ct", "pet+ct"],
            },       
            'optimizer': {
                'values': ['Adam', 'SGD']
            },
            'l2_reg': {
                'distribution': 'uniform',
                'min': 1e-6,
                'max': 1e-1
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 1e-3,
                'max': 1e-1
            },
        }
        
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'test_auc', 
                'goal': 'maximize'
            },
            'parameters': parameter_dict
        }

        # Start the sweep
        project_name = config["backbone"]+str(np.random.randint(1e8)) if config["project_name"] is None else config["project_name"]
        sweep_id = wandb.sweep(sweep_config, project=project_name) 
        wandb.agent(sweep_id, function=main)
    else:
        main(config)