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

# import monai
import nibabel as nib
from seed import seed


class KneeDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = torch.FloatTensor(labels)
        # temp = []
        # for i in range(len(labels.index)):
        #     temp.append(labels.iloc[i])
        # self.labels = torch.from_numpy(np.array(temp))

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

        # struct_arr = (struct_arr - np.array(kvs['mean'])[..., None]) / (
        #               np.array(kvs['std'])[..., None] + 1e-10)  # 1ch

        # struct_arr = struct_arr[None]  #1 ch
        # struct_arr = torch.FloatTensor(struct_arr)  #3 ch
        struct_arr = torch.from_numpy(np.array(struct_arr)).float()  # Aug

        return struct_arr, label


    def image_shape(self):
        a = self.filenames.iloc[0].to_frame().T
        mri = utils.load_mri(self.filenames.iloc[0].to_frame().T)
        # mri = mri[None]  # 1 ch
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
    # metadata = pd.read_csv(os.path.join(csv_dir, 'OAI_' + tissue + lm + '_imgs_metadata.csv'))
    metadata = pd.read_csv(os.path.join(csv_dir, 'OAI_bl_scaled_All_{}_imgs_metadata.csv'.format(lm)))
    # metadata = pd.read_csv(os.path.join(csv_dir, 'OAI_bl_normalized_All_{}_imgs_metadata.csv'.format(lm)))
    a = metadata['imgs'][0]

    '''
    corrupted_segs = [9510943, 9932578, 9800285, 9818478, 9526396, 9459743, 9176485]
    # corr_sides = ['L', 'L', 'L', 'R', 'R', 'R', 'R']
    corr_sides = ['R', 'R', 'R', 'L', 'L', 'L', 'L']
    for i in range(len(corrupted_segs)):
        a = metadata[(metadata['ID'] == corrupted_segs[i]) & (metadata['SIDE'] == corr_sides[i])].index
        metadata = metadata.drop(metadata[(metadata['ID'] == corrupted_segs[i]) & (metadata['SIDE'] == corr_sides[i])].index)
        metadata.reset_index(inplace=True, drop=True)
    '''
    # b = ''
    # for i in range(len(metadata.index)):
    #     if metadata['SIDE'].iloc[i] == 1:
    #         side = 'R'
    #     else:
    #         side = 'L'
    #
    #     a = metadata['imgs'].iloc[i]
    #
    #     b = '../data/Cropped_All/' + str(metadata['ID'].iloc[i]) + '/' + side +'/' + '001.png,../data/Cropped_All/' + str(metadata['ID'].iloc[i]) + '/' + side +'/' + '002.png,' + a
    #     c = b + ',../data/Cropped_All/' + str(metadata['ID'].iloc[i]) + '/' + side +'/' + '157.png,../data/Cropped_All/' + str(metadata['ID'].iloc[i]) + '/' + side +'/' + '158.png'
    #
    #     metadata['imgs'].iloc[i] = c


    if target_comp != 'All' and 'multi_label' not in target_comp:
        metadata = multilabel_target(metadata, target_comp)
        del metadata['Target']  # FL
        metadata['Target'] = 0  # FL
        # metadata['Target'].iloc[metadata[metadata['prog_' + target_comp] > 0].index] = 1
        # metadata['Target'] = metadata[target_comp]  # multiclass
        # metadata[target_comp].iloc[metadata[metadata[target_comp] == 1].index] = 0
        # metadata['Target'].iloc[metadata[metadata[target_comp] == 1].index] = 0
        metadata['Target'].iloc[metadata[metadata[target_comp] > 0].index] = 1  # FL


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
    # categorical_vars = ['prog_TL', 'prog_FL', 'prog_FM', 'prog_TM']
    # categorical_vars = ['TL', 'FL', 'FM', 'TM']
    # categorical_vars = ['FM', 'TM']
    categorical_vars = [target_comp]
    # categorical_vars = list(df.filter(regex='prog_'))
    if target_comp == 'b_multi_label':
        for i in categorical_vars:
            df[i].loc[df[df[i] > 0].index] = 1
    else:
        print('alaki')
        # for i in categorical_vars:
        #     df[i].loc[df[df[i] > 0].index] = 1
        # tmp_df = multi_hot(df[categorical_vars])
        # for i in categorical_vars:
        #     df[i] = tmp_df[i]

    one_hot_encoder = OneHotEncoder(sparse=False, drop="first")
    encoder_vars_array = one_hot_encoder.fit_transform(df[categorical_vars])
    encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
    encoder_vars_df = pd.DataFrame(encoder_vars_array, columns=encoder_feature_names, dtype=int)
    X_new = pd.concat([df.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1)
    # df['Target'] = [x for x in torch.from_numpy(df[categorical_vars].to_numpy())]

    if target_comp == 'b_multi_label':
        df['Target'] = [x for x in df[categorical_vars].astype(int).to_numpy()]
    elif target_comp == 'fmtm' or target_comp == 'fltl':
        df['Target'] = 0
        for i in categorical_vars:
            df.loc[df[i] >= 1, 'Target'] = 1
    else:
        # df['Target'] = [x for x in encoder_vars_df.to_numpy()]
        df['Target'] = [x for x in df[categorical_vars].to_numpy()]  # multilabel

    y = df[categorical_vars].to_numpy()
    y_for_stratif = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    df['y_stratif'] = y_for_stratif

    return df


def print_df_stats(df, df_train, df_val, df_test):
    headers = ['Images', '-> prog', '-> non-prog', 'Patients', '-> prog', '-> non-prog']

    def get_stats(df):
        df_ad = df[df['Target'] == 1]
        df_cn = df[df['Target'] == 0]
        return [len(df), len(df_ad), len(df_cn), len(df['ID'].unique()), len(df_ad['ID'].unique()),
                len(df_cn['ID'].unique())]

    stats = []
    stats.append(['All'] + get_stats(df))
    stats.append(['Train'] + get_stats(df_train))
    stats.append(['Val'] + get_stats(df_val))
    stats.append(['Test'] + get_stats(df_test))

    print(tabulate(stats, headers=headers))   
    
    print('## Unique subjects', np.unique(df.ID).shape[0])
    print('## Males', np.unique(df[df.SEX == 1].ID).shape[0])
    print('## Males FM 0', np.unique(df[(df.SEX == 1) & (df.FM == 0)].ID).shape[0])
    print('## Males FM 1', np.unique(df[(df.SEX == 1) & (df.FM == 1)].ID).shape[0])
    print('## Males FM 2', np.unique(df[(df.SEX == 1) & (df.FM == 2)].ID).shape[0])
    print('## Males FM 3', np.unique(df[(df.SEX == 1) & (df.FM == 3)].ID).shape[0])
    print('## Males FL 0', np.unique(df[(df.SEX == 1) & (df.FL == 0)].ID).shape[0])
    print('## Males FL 1', np.unique(df[(df.SEX == 1) & (df.FL == 1)].ID).shape[0])
    print('## Males FL 2', np.unique(df[(df.SEX == 1) & (df.FL == 2)].ID).shape[0])
    print('## Males FL 3', np.unique(df[(df.SEX == 1) & (df.FL == 3)].ID).shape[0])
    print('## Males TM 0', np.unique(df[(df.SEX == 1) & (df.TM == 0)].ID).shape[0])
    print('## Males TM 1', np.unique(df[(df.SEX == 1) & (df.TM == 1)].ID).shape[0])
    print('## Males TM 2', np.unique(df[(df.SEX == 1) & (df.TM == 2)].ID).shape[0])
    print('## Males TM 3', np.unique(df[(df.SEX == 1) & (df.TM == 3)].ID).shape[0])
    print('## Males TL 0', np.unique(df[(df.SEX == 1) & (df.TL == 0)].ID).shape[0])
    print('## Males TL 1', np.unique(df[(df.SEX == 1) & (df.TL == 1)].ID).shape[0])
    print('## Males TL 2', np.unique(df[(df.SEX == 1) & (df.TL == 2)].ID).shape[0])
    print('## Males TL 3', np.unique(df[(df.SEX == 1) & (df.TL == 3)].ID).shape[0])

    print('## Females', np.unique(df[df.SEX == 0].ID).shape[0])
    print('## Females FM 0', np.unique(df[(df.SEX == 0) & (df.FM == 0)].ID).shape[0])
    print('## Females FM 1', np.unique(df[(df.SEX == 0) & (df.FM == 1)].ID).shape[0])
    print('## Females FM 2', np.unique(df[(df.SEX == 0) & (df.FM == 2)].ID).shape[0])
    print('## Females FM 3', np.unique(df[(df.SEX == 0) & (df.FM == 3)].ID).shape[0])
    print('## Females FL 0', np.unique(df[(df.SEX == 0) & (df.FL == 0)].ID).shape[0])
    print('## Females FL 1', np.unique(df[(df.SEX == 0) & (df.FL == 1)].ID).shape[0])
    print('## Females FL 2', np.unique(df[(df.SEX == 0) & (df.FL == 2)].ID).shape[0])
    print('## Females FL 3', np.unique(df[(df.SEX == 0) & (df.FL == 3)].ID).shape[0])
    print('## Females TM 0', np.unique(df[(df.SEX == 0) & (df.TM == 0)].ID).shape[0])
    print('## Females TM 1', np.unique(df[(df.SEX == 0) & (df.TM == 1)].ID).shape[0])
    print('## Females TM 2', np.unique(df[(df.SEX == 0) & (df.TM == 2)].ID).shape[0])
    print('## Females TM 3', np.unique(df[(df.SEX == 0) & (df.TM == 3)].ID).shape[0])
    print('## Females TL 0', np.unique(df[(df.SEX == 0) & (df.TL == 0)].ID).shape[0])
    print('## Females TL 1', np.unique(df[(df.SEX == 0) & (df.TL == 1)].ID).shape[0])
    print('## Females TL 2', np.unique(df[(df.SEX == 0) & (df.TL == 2)].ID).shape[0])
    print('## Females TL 3', np.unique(df[(df.SEX == 0) & (df.TL == 3)].ID).shape[0])

    print('## Mean Age', np.nanmean(df.AGE))
    print('## STD Age', np.nanstd(df.AGE))
    print('## Mean Age FM 0', np.nanmean(df[df.FM == 0].AGE))
    print('## Mean Age FM 1', np.nanmean(df[df.FM == 1].AGE))
    print('## Mean Age FM 2', np.nanmean(df[df.FM == 2].AGE))
    print('## Mean Age FM 3', np.nanmean(df[df.FM == 3].AGE))
    print('## Mean Age FL 0', np.nanmean(df[df.FL == 0].AGE))
    print('## Mean Age FL 1', np.nanmean(df[df.FL == 1].AGE))
    print('## Mean Age FL 2', np.nanmean(df[df.FL == 2].AGE))
    print('## Mean Age FL 3', np.nanmean(df[df.FL == 3].AGE))
    print('## Mean Age TM 0', np.nanmean(df[df.TM == 0].AGE))
    print('## Mean Age TM 1', np.nanmean(df[df.TM == 1].AGE))
    print('## Mean Age TM 2', np.nanmean(df[df.TM == 2].AGE))
    print('## Mean Age TM 3', np.nanmean(df[df.TM == 3].AGE))
    print('## Mean Age TL 0', np.nanmean(df[df.TL == 0].AGE))
    print('## Mean Age TL 1', np.nanmean(df[df.TL == 1].AGE))
    print('## Mean Age TL 2', np.nanmean(df[df.TL == 2].AGE))
    print('## Mean Age TL 3', np.nanmean(df[df.TL == 3].AGE))

    print('## STD FM 0 ', np.nanstd(df[df.FM == 0].AGE))
    print('## STD FM 1 ', np.nanstd(df[df.FM == 1].AGE))
    print('## STD FM 2 ', np.nanstd(df[df.FM == 2].AGE))
    print('## STD FM 3 ', np.nanstd(df[df.FM == 3].AGE))
    print('## STD FL 0 ', np.nanstd(df[df.FL == 0].AGE))
    print('## STD FL 1 ', np.nanstd(df[df.FL == 1].AGE))
    print('## STD FL 2 ', np.nanstd(df[df.FL == 2].AGE))
    print('## STD FL 3 ', np.nanstd(df[df.FL == 3].AGE))
    print('## STD TM 0 ', np.nanstd(df[df.TM == 0].AGE))
    print('## STD TM 1 ', np.nanstd(df[df.TM == 1].AGE))
    print('## STD TM 2 ', np.nanstd(df[df.TM == 2].AGE))
    print('## STD TM 3 ', np.nanstd(df[df.TM == 3].AGE))
    print('## STD TL 0 ', np.nanstd(df[df.TL == 0].AGE))
    print('## STD TL 1 ', np.nanstd(df[df.TL == 1].AGE))
    print('## STD TL 2 ', np.nanstd(df[df.TL == 2].AGE))
    print('## STD TL 3 ', np.nanstd(df[df.TL == 3].AGE))

    print('## Mean BMI', np.nanmean(df.BMI))
    print('## STD BMI', np.nanstd(df.BMI))
    print('## Mean BMI FM 0', np.nanmean(df[df.FM == 0].BMI))
    print('## Mean BMI FM 1', np.nanmean(df[df.FM == 1].BMI))
    print('## Mean BMI FM 2', np.nanmean(df[df.FM == 2].BMI))
    print('## Mean BMI FM 3', np.nanmean(df[df.FM == 3].BMI))
    print('## Mean BMI FL 0', np.nanmean(df[df.FL == 0].BMI))
    print('## Mean BMI FL 1', np.nanmean(df[df.FL == 1].BMI))
    print('## Mean BMI FL 2', np.nanmean(df[df.FL == 2].BMI))
    print('## Mean BMI FL 3', np.nanmean(df[df.FL == 3].BMI))
    print('## Mean BMI TM 0', np.nanmean(df[df.TM == 0].BMI))
    print('## Mean BMI TM 1', np.nanmean(df[df.TM == 1].BMI))
    print('## Mean BMI TM 2', np.nanmean(df[df.TM == 2].BMI))
    print('## Mean BMI TM 3', np.nanmean(df[df.TM == 3].BMI))
    print('## Mean BMI TL 0', np.nanmean(df[df.TL == 0].BMI))
    print('## Mean BMI TL 1', np.nanmean(df[df.TL == 1].BMI))
    print('## Mean BMI TL 2', np.nanmean(df[df.TL == 2].BMI))
    print('## Mean BMI TL 3', np.nanmean(df[df.TL == 3].BMI))

    print('## STD FM 0 ', np.nanstd(df[df.FM == 0].BMI))
    print('## STD FM 1 ', np.nanstd(df[df.FM == 1].BMI))
    print('## STD FM 2 ', np.nanstd(df[df.FM == 2].BMI))
    print('## STD FM 3 ', np.nanstd(df[df.FM == 3].BMI))
    print('## STD FL 0 ', np.nanstd(df[df.FL == 0].BMI))
    print('## STD FL 1 ', np.nanstd(df[df.FL == 1].BMI))
    print('## STD FL 2 ', np.nanstd(df[df.FL == 2].BMI))
    print('## STD FL 3 ', np.nanstd(df[df.FL == 3].BMI))
    print('## STD TM 0 ', np.nanstd(df[df.TM == 0].BMI))
    print('## STD TM 1 ', np.nanstd(df[df.TM == 1].BMI))
    print('## STD TM 2 ', np.nanstd(df[df.TM == 2].BMI))
    print('## STD TM 3 ', np.nanstd(df[df.TM == 3].BMI))
    print('## STD TL 0 ', np.nanstd(df[df.TL == 0].BMI))
    print('## STD TL 1 ', np.nanstd(df[df.TL == 1].BMI))
    print('## STD TL 2 ', np.nanstd(df[df.TL == 2].BMI))
    print('## STD TL 3 ', np.nanstd(df[df.TL == 3].BMI))

    print('## Knees', df.ID.shape[0])

    print('## Knees non-progressors', (df.Target.values == 0).sum())
    print('## Knees progressors', (df.Target.values > 0).sum())

    print('############ Testset properties ############')

    print('## Unique subjects', np.unique(df_test.ID).shape[0])
    print('## Males', np.unique(df_test[df_test.SEX == 1].ID).shape[0])
    print('## Females', np.unique(df_test[df_test.SEX == 0].ID).shape[0])

    print('## Mean Age', np.nanmean(df_test.AGE))
    print('## STD Age', np.nanstd(df_test.AGE))

    print('## Mean BMI', np.nanmean(df_test.BMI))
    print('## STD BMI', np.nanstd(df_test.BMI))

    print('## Knees', df_test.ID.shape[0])

    print('## Knees non-progressors', (df_test.Target.values == 0).sum())
    print('## Knees progressors', (df_test.Target.values > 0).sum())

    # print('## Knees (left non-progressors)',
    #       df[(df.SIDE == 'L') & (df.Target.values == 0)].ID.shape[0])
    # print('## Knees (right non-progressors)',
    #       df[(df.SIDE == 'R') & (df.Target.values == 0)].ID.shape[0])
    #
    # print('## Knees (left progressors)',
    #       df[(df.SIDE == 'L') & (df.Target.values > 0)].ID.shape[0])
    # print('## Knees (right progressors)',
    #       df[(df.SIDE == 'R') & (df.Target.values > 0)].ID.shape[0])

    # print('## Knees non-progressors (males)', (df[df.SEX == 1].Target.values == 0).sum())
    # print('## Knees progressors (males)', (df[df.SEX == 1].Target.values > 0).sum())
    #
    # print('## Knees non-progressors (females)', (df[df.SEX == 0].Target.values == 0).sum())
    # print('## Knees progressors (females)', (df[df.SEX == 0].Target.values > 0).sum())


def build_datasets(metadata, patients_train, patients_val, patients_test, print_stats=True, normalize=True):
    kvs = GlobalKVS()
    args = parse_args()
    seed.set_ultimate_seed()

    fold_id = kvs['cur_fold']

    if print_stats:
        print_df_stats(metadata, patients_train, patients_val, patients_test)

    ################# Augmenation #################
    one_patient = metadata.iloc[0].to_frame().T
    slices = utils.load_mri(one_patient)
    transfs = []

    # if args.aug == True:
    #     transfs.extend([
    #         # monai.transforms.RandRotate(prob=1., range_x=[.05, .05]),
    #         RandomCrop(output_size=(slices.shape[2] - 10, slices.shape[3] - 10)),
    #         # monai.transforms.CenterScaleCrop(roi_scale=[1, 0.9, 1.2]),
    #         NumpyToTensor(),
    #         PTRotate3DInSlice(degree_range=[-10., 10.], prob=.5),
    #         Scale(ratio_range=(0.9, 1.2), prob=.5),
    #     ])

    if args.aug == True:
        transfs.extend([
            RandomCrop(output_size=(slices.shape[2] - 10, slices.shape[3] - 5)),
            NumpyToTensor(),
            PTRotate3DInSlice(degree_range=[-10., 10.], prob=0.5),
            # Scale(ratio_range=(0.9, 1.2), prob=0.5),
        ])

    train_dataset = KneeDataset(patients_train, np.array(patients_train['Target'], dtype=int), transform=transfs)
    val_dataset = KneeDataset(patients_val, np.array(patients_val['Target'], dtype=int))

    # train_dataset = KneeDataset(patients_train, patients_train['Target'], transform=transfs)
    # val_dataset = KneeDataset(patients_val, patients_val['Target'])

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
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
    #                         pin_memory=torch.cuda.is_available())
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


def sample_nifti(df_meta, type='general'):
    args = parse_args()

    patient, side = df_meta.ID.astype(str), df_meta.SIDE
    # spacings = (0.73, 0.73, 0.7)
    spacings = (0.3, 0.3, 0.7)

    dir_sample = os.path.join(args.output_dir, 'sample/healthy (FL)')
    if not os.path.exists(dir_sample):
        os.mkdir(dir_sample)

    if type == 'general':
        general_masks_dir = (list(df_meta.imgs.split(","))[0].replace('../data/scaled_All/lateral/', ''))[:-7]
        df_meta['imgs'] = dir_sample + '/' + general_masks_dir

        stack = []
        for i in sorted(glob.glob(df_meta['imgs'] + '/*.png')):
            slc = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            # slc = cv2.resize(slc, (int(slc.shape[1] / 2), int(slc.shape[0] / 2)))
            slc[slc == 2] = 0
            slc[slc == 4] = 0
            slc[slc == 5] = 0
            slc[slc == 6] = 0
            slc[slc != 0] = 1

            stack.append(slc)
    else:
        stack = []
        a = (df_meta.imgs)[0]
        for i in list((df_meta.imgs)[0].split(",")):# list(df_meta.imgs.split(",")):
            i = i.replace("'", '')
            i = i.replace(" ", '')
            slc = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

            print(i)
            slc = cv2.resize(slc, (int(slc.shape[1] / 2), int(slc.shape[0] / 2)))
            stack.append(slc)

    img = np.stack(stack, axis=2)
    img = img * 80

    # Save to NIfTI
    stack = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
    affine = np.diag([-1., -1., -1., 1.]).astype(np.float)
    if spacings is not None:
        affine[0, 0] = -spacings[2]
        affine[1, 1] = -spacings[1]
        affine[2, 2] = -spacings[0]

    scan = nib.Nifti1Image(stack, affine=affine)
    nib.save(scan, os.path.join(dir_sample, f'{patient}.nii'))
    print()

# metadata = pd.DataFrame(columns=['ID', 'SIDE', 'imgs'])
#
# a = str(sorted(glob.glob('../data/ind_Cropped_All/medial/9114036/R' + '/*.png'))).replace('[', '').replace(']', '')
# metadata = metadata.append(
#                         {'ID': '9114036', 'SIDE': 'R', 'imgs': a},
#                         ignore_index=True)
#
# sample_nifti(metadata, type='model_input')
