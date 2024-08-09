import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import torch
from sklearn.model_selection import StratifiedKFold

from src.datagen import ProstateData
from src.utils import create_run, loop, set_seed, get_dataloaders, get_model
import wandb
pd.options.mode.chained_assignment = None


def main(config=None):
    set_seed(config["seed"])
    
    config = create_run(config)
    if os.path.isfile(os.path.join(config["model_dir"], "cv_output.csv")):
        print("Model directory already exists. Exiting.")
        sys.exit(0)

    config['train_file'] = config["data_folder"]+'training.csv'
    config['test_file'] = config["data_folder"]+'testing.csv'
    
    # Prepare clinical data and radiomics features
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
    
    cv_output = []
    test_preds = None
    for cv, (train_idx, val_idx) in enumerate(skf.split(train_df[["idx"]], train_df["nod"])):
        cv += 1
        model, optimizer, criterion, scheduler = get_model(config)
        print("Starting cv fold ", cv)
        if cv == 1:
            print(model)
            
        # Load clinical data and normalize
        if config["cli_size"] != 0:
            train_df, test_df = data.preprocess(train_df, test_df, train_idx, config)

        if config["debugging"]:
            train_idx = list(range(20))
            val_idx = list(range(20, 40))
        train_loader, val_loader, test_loader = get_dataloaders(config, train_df, train_idx, val_idx, test_df)

        log = None
        trigger = 0
        best_loss = np.inf
        best_log = None

        for epoch in range(config['epochs']):
            print('Epoch [%d/%d]' % (epoch+1, config['epochs']))

            train_log = loop(config, train_loader, model, criterion, optimizer)
            print("Train: ", end="")
            for idx, (k, v) in enumerate(train_log.items()):
                ending = "\n" if idx == len(train_log)-1 else " - "
                print(k, round(v, 4), end=ending)

            val_log = loop(config, val_loader, model, criterion, training=False)
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

            if epoch > 3:
                trigger += 1
                if val_log["loss"] < best_loss and val_log["acc"] > .7 and val_log["auc"] > .7:
                    torch.save(model.state_dict(), os.path.join(config["model_dir"], f"bestmodel{cv}.pth"))
                    best_loss = val_log["loss"]
                    
                    best_log_keys = ["best_val_"+k for k, v in val_log.items()]
                    best_log = [v for k, v in val_log.items()]
                    print("=> saved model with the best loss")
                    trigger = 0

                elif config['early_stopping'] >= 0:
                    print('early stopping count: ', trigger, '/', config['early_stopping'])

                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

            torch.cuda.empty_cache()
        
        if best_log is None:
            best_log_keys = ["best_val_"+k for k, v in val_log.items()]
            best_log = [v for k, v in val_log.items()]
            torch.save(model.state_dict(), os.path.join(config["model_dir"], f"bestmodel{cv}.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(config["model_dir"], f"bestmodel{cv}.pth")))
        test_log, preds = loop(config, test_loader, model, criterion, training=False, return_preds=True)
        if test_preds is None:
            test_preds = pd.DataFrame(np.array(preds).T, columns=["idx", "target", f"predictions{cv}"])
        else:
            test_preds[f"predictions{cv}"] = preds[-1]
        
        print("Test results:")
        for k, v in test_log.items():
            best_log.append(v)
            best_log_keys.append("test_"+k)
            print("\t", k, ": ", round(v, 4))
        
        if config["wandb"]:
            wandb.log({k: v for k, v in list(zip(best_log_keys, best_log))})

        else:
            cv_output.append(best_log)
            pd.DataFrame(cv_output, columns=best_log_keys).to_csv(os.path.join(config["model_dir"], "cv_output.csv"), index=False)
        
        if config["cross_validation"] < 0:
            break
        
        torch.cuda.empty_cache()

    test_preds.to_csv(os.path.join(config["model_dir"], f"test_predictions.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the deep learning model training script.")
    
    parser.add_argument('--run_name', default="run", help="Name of the run (default: 'run')")
    parser.add_argument('--wandb', default=False, action="store_true", help="Log metrics to Weights and Biases (default: False)")
    parser.add_argument('--debugging', default=False, action="store_true", help="Enable debugging mode (default: False)")
    parser.add_argument('--run_config_file', default=None, help="Path to the run configuration file (default: None)")
    parser.add_argument('--project_name', default=None, help="Name of the project for wandb(default: None)")
    parser.add_argument('--data_folder', default="./data/", help="Folder containing the data (default: './data/')")
    parser.add_argument('--image_folder', default="./data/ProstateData/", help="Folder containing the image data (default: './data/ProstateData/')")
    parser.add_argument('--results_folder', default="./results/", help="Folder to save the results")
    parser.add_argument('--load_from', default=None, help="Path to a saved model weights file to load (default: None)")
    parser.add_argument('--seed', default=58, type=int, help="Random seed for reproducibility (default: 58)")

    parser.add_argument('--class_weight', default=2, type=float, help="Class weight for the loss function (default: 2)")
    parser.add_argument('--image_size', default=64, type=int, help="Size of the input images (default: 256)")
    
    parser.add_argument('--activation', default='leaky_relu', choices=["relu", "leaky_relu", "gelu"], help="Activation function to use (default: 'leaky_relu')")
    parser.add_argument('--fe1_depth', default=2, type=int, help="Depth of the first feature extractor (default: 2)")
    parser.add_argument('--fusion_method', default="multiply", choices=["concat", "adaptive", "pet_weighted", "ct_weighted", "average", "multiply"], help="Method for fusing features (default: 'average')")
    parser.add_argument('--mm_fusion', default="adaptive", choices=["concat", "adaptive", "multiply", "summation"], help="Method for multimodal fusion (default: 'concat')")
    parser.add_argument('--image_fusion', default="mid", choices=["early", "mid", "late"], help="Stage at which image data is fused (default: 'early')")
    parser.add_argument('--backbone', default="small", choices=["small", "resnet50", "resnet18", "densenet", "mobilenet", "efficientnet", "efficientnetb1", "mnasnet", "vgg", "shufflenet"], help="Backbone network to use (default: 'small')")
    parser.add_argument('--pretrained', default=True, action='store_false', help="Use pretrained weights for the backbone (default: True)")
    parser.add_argument('--clinical_group', default="clinical+pet", choices=["None", "clinical", "clinical+radiomics", "clinical+pet", "clinical+ct", "pet", "ct", "radiomics"], help="Clinical data group to use (default: 'clinical+pet')")
    parser.add_argument('--image_data', default="pet+ct", choices=["pet", "ct", "pet+ct", "None"], help="Type of image data to use (default: 'pet+ct')")

    parser.add_argument('--transform', default=True, action='store_false', help="Apply transformations to the data (default: True)")
    parser.add_argument('--crop', default=True, action='store_false', help="Apply cropping to the images (default: True)")
    parser.add_argument('--normalization', default=True, action='store_false', help="Apply normalization to the images (default: True)")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size for training (default: 8)")
    parser.add_argument('--val_batch_size', default=8, type=int, help="Batch size for validation (default: 8)")

    parser.add_argument('--epochs', default=100, type=int, help="Number of training epochs (default: 30)")
    parser.add_argument("--cross_validation", default=5, type=int, help="Number of cross-validation folds (default: 5)")
    
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'], help="Optimizer to use (default: 'SGD')")
    parser.add_argument('--learning_rate', default=0.027, type=float, help="Learning rate for the optimizer (default: 0.027)")
    parser.add_argument('--lr_scheduler', default="ReduceLROnPlateau", choices=['CosineAnnealingLR', 'ReduceLROnPlateau', "ConstantLR"], help="Learning rate scheduler to use (default: 'ReduceLROnPlateau')")
    parser.add_argument('--l2_reg', default=0.027, type=float, help="L2 regularization strength (default: 0.027)")
    parser.add_argument('--early_stopping', default=20, type=int, help="Early stopping patience (default: 10)")
    
    config = parser.parse_args()

    return config


if __name__ == '__main__':
    config = vars(parse_args())
    if config["run_config_file"]:
        new_run_name = config["run_name"]
        load_from = config["load_from"]
        wandb_flag = config["wandb"]
        cross_validation = config["cross_validation"]
        with open(config["run_config_file"], "r") as f:
            config.update(json.load(f))
        config.update({
            "run_name": new_run_name,
            "load_from": load_from,
            "wandb": wandb_flag,
            "cross_validation": cross_validation,
        })
    main(config=config)
