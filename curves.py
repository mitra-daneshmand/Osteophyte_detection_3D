import os.path
from operator import itemgetter

import matplotlib.pyplot as plt
import torch

from kvs import GlobalKVS
from args import parse_args


def plot_learning_curve(trainer, plt_title=None):
    kvs = GlobalKVS()
    args = parse_args()
    fold_id = kvs['cur_fold']

    fig, axes = plt.subplots(4, sharex=True, figsize=(20, 40))

    metrics = list(map(itemgetter(0), trainer.history.epoch_metrics.items()))

    epochs = range(1, len(trainer.history[metrics[2]]) + 1)


    #########################  AP  #########################
    train_aps = [torch.tensor(item).cpu().numpy() for item in trainer.ap]
    val_aps = [torch.tensor(item).cpu().numpy() for item in trainer.eval_ap]

    plt.title(plt_title)
    plt.sca(axes[0])
    plt.grid()
    plt.plot(epochs, train_aps, 'b-', label='Train')
    plt.plot(epochs, val_aps, 'b--', label='Val')

    plt.ylabel('Average Precision Scores', fontsize=20)
    plt.ylim(0, 1)
    plt.legend(prop={"size": 20})


    #######################  Accuracy  #######################
    train_bal_acc = trainer.history['acc']  # acc_metric
    val_bal_acc = trainer.history['val_acc']  # val_acc_metric

    plt.sca(axes[1])
    plt.grid()
    plt.plot(epochs, train_bal_acc, 'g-', label='Train')
    plt.plot(epochs, val_bal_acc, 'g--', label='Val')

    plt.ylabel('Balanced Accuracy', fontsize=20)
    plt.ylim(0, 1)
    # plt.ylim(0, 100)  # multiclass
    plt.legend(prop={"size": 20})


    ########################  Loss  ########################
    plt.sca(axes[2])
    plt.grid()

    plt.plot(epochs, trainer.history['loss'], 'r-', label='Train')
    plt.plot(epochs, trainer.history['val_loss'], 'r--', label='Val')

    plt.ylabel('Loss', fontsize=20)
    plt.ylim(0, 1)
    # plt.ylim(0, 100)  # multiclass
    plt.legend(prop={"size": 20})


    #######################  ROC_AUC  #######################
    plt.title(plt_title)
    plt.sca(axes[3])
    plt.grid()
    plt.plot(epochs, trainer.roc_auc, 'g-', label='Train')
    plt.plot(epochs, trainer.eval_roc_auc, 'g--', label='Val')

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('ROC_AUC Scores', fontsize=20)
    plt.ylim(0, 1)
    plt.legend(prop={"size": 20})

    plt.savefig(os.path.join(args.output_dir, 'Metrics_fold{}.pdf').format(fold_id), bbox_inches='tight')
    plt.show()

