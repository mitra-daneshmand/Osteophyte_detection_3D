import os
import gc
import glob

import numpy as np
import pandas as pd
import cv2
import torch
from tabulate import tabulate

from augmenttion import NumpyToTensor, RandomCrop, PTRotate3DInSlice, Scale

from torch.utils.data import Dataset, DataLoader

import utils
from args import parse_args
from torchsampler import imbalanced
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from kvs import GlobalKVS
from tqdm import tqdm

import nibabel as nib
from seed import seed


class KneeDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = torch.FloatTensor(labels)

        self.transform = transform

        self.num_inputs = 1
        self.num_targets = 1

        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        kvs = GlobalKVS()
        kvs.update('mean', self.mean)
        kvs.update('std', self.std)
        label = self.labels[idx].unsqueeze(-1)

        struct_arr = utils.load_mri(self.filenames.iloc[idx].to_frame().T)

        if self.transform is not None:
            for t_fn in self.transform:
                if hasattr(t_fn, "randomize"):
                    t_fn.randomize()
                struct_arr = t_fn(struct_arr)

        struct_arr = (struct_arr - np.array(kvs['mean'])[None, ..., None, None]) / (np.array(kvs['std'])[None, ..., None, None] + 1e-10)  # 3ch

        struct_arr = torch.from_numpy(np.array(struct_arr)).float()  # Aug

        return struct_arr, label


    def image_shape(self):
        a = self.filenames.iloc[0].to_frame().T
        mri = utils.load_mri(self.filenames.iloc[0].to_frame().T)
        return [len(mri), (mri[0].shape[0]), (mri[0].shape[1]), (mri[0].shape[2])]


    def fit_normalization(self):
        kvs = GlobalKVS()
        gc.collect()
        torch.cuda.empty_cache()

        image_shape = self.image_shape()

        num_sample = len(self.filenames.index)
        all_struct_arr = torch.zeros((num_sample, image_shape[0], image_shape[1], image_shape[2], image_shape[3]))

        sampled_filenames = np.random.choice(self.filenames.ID, num_sample, replace=False)
        for i, filename in enumerate(tqdm(sampled_filenames, desc='Computing mean and std values:')):
            struct_arr = utils.load_mri(self.filenames[self.filenames['ID'] == filename])
            struct_arr = torch.from_numpy(struct_arr).float()
            all_struct_arr[i] = struct_arr

        self.mean = torch.mean(all_struct_arr, axis=0)
        self.std = torch.std(all_struct_arr, axis=0)

        print("Saving mean and std files!")
        fold_id = kvs['cur_fold']
        torch.save(self.mean, os.path.join('sessions/mean/', f'mean_{kvs["tissue"]}_fold_{fold_id}.pth'))
        torch.save(self.std, os.path.join('sessions/std/', f'std_{kvs["tissue"]}_fold_{fold_id}.pth'))

        print("Normalization done!")


    def get_raw_image(self, idx):
        return utils.load_mri(self.filenames.iloc[idx])


def init_metadata(tissue, lm, target_comp, csv_dir):
    metadata = pd.read_csv(os.path.join(csv_dir, 'OAI_bl_scaled_All_{}_imgs_metadata.csv'.format(lm)))

    if target_comp != 'All' and 'multi_label' not in target_comp:
        metadata = multilabel_target(metadata, target_comp)
        del metadata['Target']  # FL
        metadata['Target'] = 0  # FL
        metadata['Target'].iloc[metadata[metadata[target_comp] > 0].index] = 1


    elif 'multi_label' in target_comp:
        metadata = multilabel_target(metadata, target_comp)
    else:
        metadata = multilabel_target(metadata, target_comp)
        del metadata['Target']
        metadata.rename(columns={'Target_4y': 'Target'}, inplace=True)

    return metadata


def multi_hot(df):
    max_tlg = df['TL'].iloc[np.argmax(df['TL'])]

    inds = df[df['FL'] == 0].index
    df['FL'].loc[inds] = 4
    inds = df[df['FL'] == 1].index
    df['FL'].loc[inds] = 5
    inds = df[df['FL'] == 2].index
    df['FL'].loc[inds] = 6
    inds = df[df['FL'] == 3].index
    df['FL'].loc[inds] = 7

    inds = df[df['FM'] == 0].index
    df['FM'].loc[inds] = 8
    inds = df[df['FM'] == 1].index
    df['FM'].loc[inds] = 9
    inds = df[df['FM'] == 2].index
    df['FM'].loc[inds] = 10
    inds = df[df['FM'] == 3].index
    df['FM'].loc[inds] = 11

    inds = df[df['TM'] == 0].index
    df['TM'].loc[inds] = 12
    inds = df[df['TM'] == 1].index
    df['TM'].loc[inds] = 13
    inds = df[df['TM'] == 2].index
    df['TM'].loc[inds] = 14
    inds = df[df['TM'] == 3].index
    df['TM'].loc[inds] = 15

    return df


def multilabel_target(df, target_comp):
    categorical_vars = [target_comp]
    if target_comp == 'b_multi_label':
        for i in categorical_vars:
            df[i].loc[df[df[i] > 0].index] = 1
    else:
        print('')

    one_hot_encoder = OneHotEncoder(sparse=False, drop="first")
    encoder_vars_array = one_hot_encoder.fit_transform(df[categorical_vars])
    encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
    encoder_vars_df = pd.DataFrame(encoder_vars_array, columns=encoder_feature_names, dtype=int)
    X_new = pd.concat([df.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1)

    if target_comp == 'b_multi_label':
        df['Target'] = [x for x in df[categorical_vars].astype(int).to_numpy()]
    elif target_comp == 'fmtm' or target_comp == 'fltl':
        df['Target'] = 0
        for i in categorical_vars:
            df.loc[df[i] >= 1, 'Target'] = 1
    else:
        df['Target'] = [x for x in df[categorical_vars].to_numpy()]  # multilabel

    y = df[categorical_vars].to_numpy()
    y_for_stratif = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    df['y_stratif'] = y_for_stratif

    return df


def build_datasets(metadata, patients_train, patients_val, patients_test, normalize=True):
    kvs = GlobalKVS()
    args = parse_args()
    seed.set_ultimate_seed()

    fold_id = kvs['cur_fold']

    ################# Augmenation #################
    one_patient = metadata.iloc[0].to_frame().T
    slices = utils.load_mri(one_patient)
    transfs = []

    if args.aug == True:
        transfs.extend([
            RandomCrop(output_size=(slices.shape[2] - 10, slices.shape[3] - 5)),
            NumpyToTensor(),
            PTRotate3DInSlice(degree_range=[-10., 10.], prob=0.5),
        ])

    train_dataset = KneeDataset(patients_train, np.array(patients_train['Target'], dtype=int), transform=transfs)
    val_dataset = KneeDataset(patients_val, np.array(patients_val['Target'], dtype=int))

    if normalize:
        if not os.path.exists(os.path.join(args.output_dir + 'mean/', f'mean_{args.tissue}{args.lm}_fold_{kvs["cur_fold"]}.pth')):
            print('Calculating mean and std for normalization:')
            mean, std = estimate_mean_std(train_dataset)
            kvs.update('mean', mean)
            kvs.update('std', std)
            train_dataset.mean, train_dataset.std = mean, std
            val_dataset.mean, val_dataset.std = mean, std

            print("Saving mean and std files!")
            torch.save(mean, os.path.join(args.output_dir + 'mean/', f'mean_{args.tissue}{args.lm}_fold_{fold_id}.pth'))
            torch.save(std, os.path.join(args.output_dir + 'std/', f'std_{args.tissue}{args.lm}_fold_{fold_id}.pth'))

            print("Normalization done!")
        else:
            print('Loading mean and std for normalization:')
            train_dataset.mean = torch.load(
                os.path.join(args.output_dir + 'mean/', f'mean_{args.tissue}{args.lm}_fold_{fold_id}.pth'))
            train_dataset.std = torch.load(
                os.path.join(args.output_dir + 'std/', f'std_{args.tissue}{args.lm}_fold_{fold_id}.pth'))
            val_dataset.mean, val_dataset.std = train_dataset.mean, train_dataset.std
            kvs.update('mean', train_dataset.mean)
            kvs.update('std', train_dataset.std)
    else:
        print('Dataset is not normalized, this could dramatically decrease performance')

    return train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset):
    args = parse_args()

    train_loader = DataLoader(train_dataset, sampler=imbalanced.ImbalancedDatasetSampler(train_dataset),
                              batch_size=args.batch_size,
                              num_workers=8, pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def estimate_mean_std(train_dataset):
    mean_std_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, pin_memory=torch.cuda.is_available())
    means = []
    stds = []
    for sample in tqdm(mean_std_loader, desc='Computing mean and std values:'):
        local_batch, local_labels = sample

        means.append(torch.mean(local_batch, dim=(1, 3, 4)).cpu().numpy())
        stds.append(torch.std(local_batch, dim=(1, 3, 4)).cpu().numpy())

    mean = np.mean(np.concatenate(means, axis=0), axis=0)
    std = np.mean(np.concatenate(stds, axis=0), axis=0)

    return mean, std

