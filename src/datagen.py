import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchio as tio


class ProstateImages(torch.utils.data.Dataset):
    def __init__(self, config, df, partition, split_idx=None):
        self.config = config
        self.cli_only = True if config["image_data"] is None or config["image_data"] in ["None", "none"] else False
        self.image_size = config["image_size"]
        self.img_folder = config["image_folder"]
        self.cli_size = config["cli_size"]
        self.target_var = "nod"
        if split_idx is not None:
            self.df = df.iloc[split_idx].reset_index(drop=True)
        else:
            self.df = df
        
        self.crop = config["crop"]
        self.global_norm = False
        self.pet_min, self.pet_max = None, None

        if partition == "train" and config["transform"]:
            self.augmentation = tio.transforms.OneOf({
            tio.transforms.OneOf({
                tio.transforms.RandomBiasField(): .25,
                tio.transforms.RandomGhosting(): .25,
                tio.transforms.RandomSpike(): .25,
            }): .8
        })
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pt_id = self.df["idx"].iloc[index]
        label = torch.FloatTensor(np.array(self.df[self.target_var][self.df["idx"] == pt_id].item())[np.newaxis])
        cli_data = torch.squeeze(torch.Tensor(self.df[self.df["idx"] == pt_id].drop(["idx", self.target_var], axis=1).values)) if self.cli_size > 0 else None
        if self.cli_only:
            data = [torch.zeros(1), torch.zeros(1), torch.zeros(1), cli_data]
            return data, label, pt_id
        pet_arr = self.get_img(os.path.join(self.img_folder, str(pt_id), "PET"))/255.
        ct_arr = self.get_img(os.path.join(self.img_folder, str(pt_id), "CT"))/255.
        seg_arr = self.get_img(os.path.join(self.img_folder, str(pt_id), "SEG"))
        
        if self.augmentation is not None:
            pet_arr = self.augmentation(pet_arr)
            ct_arr = self.augmentation(ct_arr)
        
        data = [pet_arr, ct_arr, seg_arr]
        if self.cli_size > 0:
            data.append(cli_data)

        return data, label, pt_id
    
    def get_img(self, path):
        img = []
        for sl_nb in sorted(os.listdir(path)):
            img.append(np.array(Image.open(os.path.join(path, sl_nb)).convert("L")))
        arr = self.resize(np.array(img))
        return torch.unsqueeze(arr, 0)

    def resize(self, data):
        image_dim = data.shape[-1]
        if image_dim != self.image_size:
            if self.crop:
                transformer = T.CenterCrop(int(image_dim/2))
                data = transformer(torch.Tensor(data))
                if int(image_dim/2) == self.image_size:
                    return data
                else:
                    transformer = T.Resize((self.image_size, self.image_size))
                    return transformer(torch.Tensor(data))
            else:
                transformer = T.Resize((self.image_size, self.image_size))
                return transformer(torch.Tensor(data))
        return torch.Tensor(data)


class ProstateData:
    def __init__(self, config):
        self.config = config

        self.target = "nod"
        self.train_ids = pd.read_csv(config["train_file"])["idx"]
        self.test_ids = pd.read_csv(config["test_file"])["idx"]

        self.label_train = pd.read_csv(config["train_file"])[self.target]
        self.label_test = pd.read_csv(config["test_file"])[self.target]
        self.train_df = pd.read_csv(config["train_file"]).drop(columns=self.target, axis=1)
        self.test_df = pd.read_csv(config["test_file"]).drop(columns=self.target, axis=1)
        self.stats = None

        self.pet_radiomics  = [col for col in self.train_df.columns if col[:4]=="pet_"]
        self.ct_radiomics  = [col for col in self.train_df.columns if col[:3]=="ct_"]
        
        # assert len(self.pet_radiomics) == 105 and len(self.ct_radiomics) == 105, "Radiomics features are missing."
        self.clinical_features = [
            'age', 'risk', 'psa', 'cT',
            'ISUP', 'gle', 'suv'
        ]

        self.feature_groups = {
            "clinical": self.clinical_features,
            "radiomics": self.pet_radiomics+self.ct_radiomics,
            "pet": self.pet_radiomics,
            "ct": self.ct_radiomics
        }
        self.groups = [
            "clinical",
            "clinical+radiomics",
            "clinical+pet",
            "clinical+ct"
        ]
        self.all_feats_train, self.all_feats_test = self.define_features()
        
    def col_types(self, df):
        cont_cols = ["age", "psa", "suv"] 
        [cont_cols.append(i) for i in self.pet_radiomics]
        [cont_cols.append(i) for i in self.ct_radiomics]

        bin_cols, cat_cols = [], []
        for col in df.columns:
            if len(pd.unique(df[col])) == 2:
                bin_cols.append(col)
            elif len(pd.unique(df[col])) > 1 and len(pd.unique(df[col])) < 10:
                cat_cols.append(col)
        
        return cont_cols, bin_cols, cat_cols

    def define_features(self, encoding=True):
        # Define columns
        self.continous_cols, self.binary_cols, self.cat_cols = self.col_types(self.train_df.drop(columns="idx"))

        if encoding:
            all_feats_train, all_feats_test = self.encode_vars()
        else:
            all_feats_train = pd.concat([self.train_features[self.cat_cols], self.train_features[self.binary_cols], self.train_features[self.continous_cols]], axis=1)
            all_feats_test = pd.concat([self.test_features[self.cat_cols], self.test_features[self.binary_cols], self.test_features[self.continous_cols]], axis=1)

        if self.label_train.isna().sum().any():
            drop_idx = all_feats_train[self.label_train.isna()]
            all_feats_train = all_feats_train.drop(drop_idx, axis=0).reset_index(drop=True)
            print("Missing labels in the train set is dropped.")
        if self.label_test.isna().sum().any():
            drop_idx = all_feats_test[self.label_test.isna()]
            all_feats_test = all_feats_test.drop(drop_idx, axis=0).reset_index(drop=True)
            print("Missing labels in the test set is dropped.")
        
        return all_feats_train, all_feats_test

    def encode_vars(self):
        dummy_cols_train = [pd.get_dummies(self.train_df[col], prefix=col) for col in self.cat_cols]
        encoded_train = pd.concat(dummy_cols_train, axis=1)

        dummy_cols_test = [pd.get_dummies(self.test_df[col], prefix=col) for col in self.cat_cols]
        encoded_test = pd.concat(dummy_cols_test, axis=1)

        add_to_test = [[col_idx, col] for col_idx, col in enumerate(encoded_train.columns) if col not in encoded_test.columns]
        if len(add_to_test) > 0:
            for i in add_to_test:
                encoded_test.insert(i[0], column=i[1], value=np.zeros(len(encoded_test)))
        
        add_to_train = [[col_idx, col] for col_idx, col in enumerate(encoded_test.columns) if col not in encoded_train.columns]
        if len(add_to_train) > 0: 
            for i in add_to_train:
                encoded_train.insert(i[0], column=i[1], value=np.zeros(len(encoded_train)))

        assert encoded_train.shape[-1] == encoded_test.shape[-1], "Error during encoding."
        
        merged_train = pd.concat([encoded_train, self.train_df[self.binary_cols], self.train_df[self.continous_cols]], axis=1)
        merged_test = pd.concat([encoded_test, self.test_df[self.binary_cols], self.test_df[self.continous_cols]], axis=1)
        [self.binary_cols.append(i) for i in encoded_train.columns]
        return merged_train, merged_test
    
    def selected_feats(self, group):
        if group is None or group in ["none", "None"]:
            selected_vars_train = pd.concat([self.train_ids, self.label_train], axis=1)
            selected_vars_test = pd.concat([self.test_ids, self.label_test], axis=1)
        else:
            cols =  self.selected_cols(group)

            selected_vars_train = pd.concat([self.train_ids, self.all_feats_train[cols], self.label_train], axis=1)
            selected_vars_test = pd.concat([self.test_ids, self.all_feats_test[cols], self.label_test], axis=1)

        return selected_vars_train, selected_vars_test

    def selected_cols(self, group):
        column_list = []
        for g in group.split("+"):
            for k, v in self.feature_groups.items():
                if g == k:
                    for i in v:
                        for col in self.all_feats_train.columns:
                            if i == col:
                                column_list.append(i)
                            elif col[-2] == "_":
                                if i in col:
                                    column_list.append(col)
        return column_list

    def preprocess(self, train_df, test_df, train_ids, config):
        self.config = config # model dir is updated.
        eps = 1e8
        
        tr_med = train_df.iloc[train_ids].median()
        tr_min = train_df.iloc[train_ids].min()
        tr_max = train_df.iloc[train_ids].max()

        for col in train_df.iloc[:, 1:-1]:
            # Fill the missing values in the total data with the median of the training set
            train_df[col] = train_df[col].fillna(tr_med[col])
            test_df[col] = test_df[col].fillna(tr_med[col])
            if self.config["normalization"]:
                # Min-max normalization of the total data with the training set values
                train_df[col] -= tr_min[col]
                train_df[col] /= (tr_max[col] - tr_min[col] + eps * (tr_max[col] == tr_min[col]))

                test_df[col] -= tr_min[col]
                test_df[col] /= (tr_max[col] - tr_min[col] + eps * (tr_max[col] == tr_min[col]))
        
        # Save training statistics to apply to the test set
        if self.stats is None:
            self.stats = pd.concat([tr_med, tr_min, tr_max], axis=1).T
        else:
            self.stats = pd.concat([self.stats, (pd.concat([tr_med, tr_min, tr_max], axis=1).T)], axis=0)
        
        os.makedirs(config["model_dir"], exist_ok=True)
        pd.DataFrame(self.stats).to_csv(os.path.join(config["model_dir"], "train_stats.csv"))
        return train_df, test_df
