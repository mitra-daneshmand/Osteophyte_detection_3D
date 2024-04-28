import glob
from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from kvs import GlobalKVS
from trainer import Trainer
from args import parse_args

from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss

import torchsample
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.metrics import BinaryAccuracy
import torchmetrics
from resnet import generate_model, generate_model_tuning, get_fine_tuning_parameters

from sklearn.metrics import balanced_accuracy_score, accuracy_score


def build_model(network, depth, learning_rate, weight_decay):
    args = parse_args()
    if network == 'ResNet':
        if args.target_comp == 'm_multi_label':
            net = MultiTaskHead((4,), (4,), model_depth=depth, num_classes=args.n_classes)
        else:
            net = generate_model(model_depth=depth, num_classes=args.n_classes)

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    if args.pretrain_path:
        print('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path, map_location=torch.device('cpu'))

        d = pretrain['state_dict']
        for k, v in list(d.items()):
            pretrain['state_dict'][k.replace('module.', '')] = pretrain['state_dict'].pop(k)

        net_dict = net.state_dict()
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        net.load_state_dict(net_dict)

        net.fc = nn.Linear(net.fc.in_features, args.n_classes)

    if torch.cuda.is_available():
        net = net.cuda()
        cuda_device = torch.cuda.current_device()
    else:
        cuda_device = -1
        print('GPU not available')
    if args.pretrain_path:
        parameters = get_fine_tuning_parameters(net, args.ft_portion)
    else:
        parameters = net.parameters()


    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    callbacks = []
    loss_function = nn.BCEWithLogitsLoss()  # Binary Classification
    # loss_function = nn.CrossEntropyLoss()  # Multi class classification
    # loss_function = MultiTaskClassificationLoss()

    trainer = Trainer(net)
    trainer.compile(loss=loss_function.cuda(),
                    optimizer=optimizer,
                    metrics=[BalancedAccuracy()],  # Binary Classification
                    # metrics=[CategoricalAccuracyWithLogits()],  # Multi class classification /or LabelBasedMacroAccuracy
                    # metrics=[accuracy()],  # multi_label
                    # metrics=[map()],  # multi_label_multi_target
                    callbacks=callbacks)

    return net, trainer, cuda_device


def train_model(trainer, train_loader, val_loader, cuda_device, num_epoch=1):
    trainer.fit_loader(train_loader,
                       val_loader=val_loader,
                       num_epoch=num_epoch,
                       verbose=1,
                       cuda_device=cuda_device)

# ------------------------ MultiTaskModel ---------------------------
class MultiTaskHead(nn.Module):
    def __init__(self, n_tasks, n_cls, model_depth, num_classes):

        super(MultiTaskHead, self).__init__()

        if isinstance(n_cls, int):
            n_cls = (n_cls, )

        if isinstance(n_tasks, int):
            n_tasks = (n_tasks,)

        assert len(n_cls) == len(n_tasks)

        self.n_tasks = n_tasks
        self.n_cls = n_cls

        classification = generate_model(model_depth=model_depth, num_classes=num_classes)
        for task_type_idx, (n_tasks, task_n_cls) in enumerate(zip(self.n_tasks, self.n_cls)):
            for task_idx in range(n_tasks):
                self.__dict__['_modules'][f'task_{task_type_idx+task_idx}'] = classification

    def forward(self, features):
        res = []
        for j in range(sum(self.n_tasks)):
            res.append(self.__dict__['_modules'][f'task_{j}'](features))
        return res

# ------------------------ Metrics ---------------------------
class BinaryAccuracyWithLogits(torchsample.metrics.BinaryAccuracy):
    def __call__(self, y_pred, y_true):
        a = (y_true.int().squeeze().detach().cpu())
        b = (F.sigmoid(y_pred).squeeze().detach().cpu()).round()
        return super(BinaryAccuracyWithLogits, self).__call__(b, a)


class CategoricalAccuracyWithLogits(torchsample.metrics.CategoricalAccuracy):
    def __call__(self, y_pred, y_true):
        a = (y_true.int().squeeze().detach().cpu())
        b = (F.softmax(y_pred).squeeze().detach().cpu())
        return super(CategoricalAccuracyWithLogits, self).__call__(b, a)


class BalancedAccuracy(torchmetrics.Metric):

    def __init__(self):
        super(BalancedAccuracy, self).__init__()
        self._name = 'acc'
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true) -> None:

        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            self.cuda(cuda_device)

        # self.b_acc = balanced_accuracy_score((np.array(y_true.int().squeeze().detach().cpu().numpy())).astype(int),
        #                                      np.array(F.sigmoid(y_pred).squeeze().detach().cpu().numpy()).round())

        self.b_acc = balanced_accuracy_score(y_true, y_pred.round())  # dl_evaluation

    def compute(self):
        return self.b_acc


class LabelBasedMacroAccuracy(torchmetrics.Metric):  # multi_calss

    def __init__(self):
        super(LabelBasedMacroAccuracy, self).__init__()
        self._name = 'acc'
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true) -> None:

        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            self.cuda(cuda_device)

        l_acc_num = torch.sum(torch.logical_and(y_true, y_pred), 0)
        l_acc_den = torch.sum(torch.logical_or(y_true, y_pred), 0)

        self.m_avg_acc = torch.mean(l_acc_num/l_acc_den).detach().cpu().numpy()

    def compute(self):
        return self.m_avg_acc


class map(torchmetrics.Metric):  # multi_label_multi_target
    def __init__(self):
        super(map, self).__init__()
        self._name = 'map'
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def patk(self, actual, pred):
        common_values = []
        for i in range(len(pred)):
            if pred[i] == actual[i]:
                common_values.append(pred[i])

        return len(common_values) / len(pred)

    def update(self, y_pred, y_true) -> None:

        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            self.cuda(cuda_device)

        average_precision = []
        for i in range(len(y_true)):
            average_precision.append(self.patk(y_true[i], y_pred[i]))

        self.map = np.mean(average_precision)
        print(self.map)

    def compute(self):
        return self.map


class MultiTaskClassificationLoss(nn.Module):
    def __init__(self):
        super(MultiTaskClassificationLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target_cls):
        loss = 0
        n_tasks = len(pred)

        for task_id in range(n_tasks):
            loss += self.cls_loss(pred[task_id], target_cls[:, task_id])

        loss /= n_tasks

        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.eps = 1e-6

    def forward(self, input, target):
        # input are not the probabilities, they are just the cnn out vector
        # input and target shape: (bs, n_classes)
        # sigmoid
        probs = torch.sigmoid(input)
        log_probs = -torch.log(probs)

        focal_loss = torch.sum(torch.pow(1 - probs + self.eps, self.gamma).mul(log_probs).mul(target), dim=1)
        # BCE_loss = torch.sum(log_probs.mul(target), dim = 1)

        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()  # , bce_loss

