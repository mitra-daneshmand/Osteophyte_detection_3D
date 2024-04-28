import gc
import glob
import os

import pandas as pd
import numpy as np
import torch

import curves
import datasets
import models
from args import parse_args

from kvs import GlobalKVS
from termcolor import colored


import warnings
warnings.filterwarnings('ignore')


def main():
    args = parse_args()
    print(colored('Config arguments', 'green'))
    print('network              :', args.network)
    print('model depth          :', args.model_depth)
    print('output directory     :', args.output_dir)
    print('no epochs            :', args.epochs)
    print('batch size           :', args.batch_size)
    print('learning rate        :', args.learning_rate)
    print('weight_decay         :', args.weight_decay)
    print('target compartment   :', args.target_comp)
    print('tissue               :', args.tissue)

    kvs = GlobalKVS()
    kvs.update('args', args)

    metadata = datasets.init_metadata(args.tissue, args.lm, args.target_comp, args.csv_dir)

    normalize = True
    save_dir = 'sessions/{}/'.format(args.target_comp)

    test_index = pd.read_csv(os.path.join(save_dir, 'test.csv'))
    train_val_index = pd.read_csv(os.path.join(save_dir, 'train_val.csv'))
    test_set, train_val_set = metadata.iloc[test_index.values.flatten()], metadata.iloc[train_val_index.values.flatten()]
    kvs.update('test_set', test_set)

    train_val_set.reset_index(inplace=True, drop=True)

    for fold_num in range(5):
        train_index = pd.read_csv(os.path.join(save_dir, 'fold_{}_train.csv'.format(str(fold_num))))
        val_index = pd.read_csv(os.path.join(save_dir, 'fold_{}_val.csv'.format(str(fold_num))))
        val_set, train_set = train_val_set.iloc[val_index.values.flatten()], train_val_set.iloc[train_index.values.flatten()]

        torch.cuda.empty_cache()

        kvs.update('cur_fold', fold_num)
        kvs.update('prev_model', None)

        df = train_set.value_counts(train_set.y_stratif)

        print(colored('====> ', 'blue') + f'Training fold {fold_num}....')

        print('Preparing datasets...')
        train_dataset, val_dataset = datasets.build_datasets(train_val_set, train_set, val_set, test_set, print_stats=False, normalize=normalize)  # metadata
        train_loader, val_loader = datasets.build_loaders(train_dataset, val_dataset)

        print('Building model and trainer...')
        net, trainer, cuda_device = models.build_model(args.network, args.model_depth, args.learning_rate, args.weight_decay)

        print('Starting training...')
        models.train_model(trainer, train_loader, val_loader, cuda_device, num_epoch=args.epochs)

        curves.plot_learning_curve(trainer, args.tissue)

        del net, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
