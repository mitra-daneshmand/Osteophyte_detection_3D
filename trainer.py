from __future__ import absolute_import
from __future__ import print_function

import operator
import os
import gc
import math
import torch
import torch as th
import torchmetrics
import torchsample
import torch.nn.functional as F

from torchsample.callbacks import CallbackContainer, TQDM, History
from torchsample.constraints import ConstraintCallback, ConstraintContainer
from torchsample.initializers import InitializerContainer
from torchsample.metrics import MetricContainer, MetricCallback
from torchsample.modules._utils import _add_regularizer_to_loss_fn, \
    _parse_num_inputs_and_targets_from_loader, _is_tuple_or_list
from torchsample.modules.module_trainer import _get_helper
from torchsample.regularizers import RegularizerCallback, RegularizerContainer
from sklearn.metrics import roc_auc_score

# import torchvision
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from kvs import GlobalKVS


class Trainer(torchsample.modules.module_trainer.ModuleTrainer):
    def __init__(self, model):
        super().__init__(model)
        self.ap = []
        self.eval_ap = []
        self.roc_auc = []
        self.eval_roc_auc = []

    def save_checkpoint(self, model, val_metric_name, comparator='gt'):
        kvs = GlobalKVS()
        fold_id = kvs['cur_fold']
        epoch = kvs['cur_epoch']
        val_metric = kvs[f'val_metrics_fold_[{fold_id}]'][val_metric_name]
        comparator = getattr(operator, comparator)
        cur_snapshot_name = os.path.join('sessions/', f'fold_{fold_id}_epoch_{epoch + 1}.pth')

        if kvs['prev_model'] is None:
            print(colored('====> ', 'green') + 'Snapshot was saved to', cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)
            torch.save(model.state_dict(), cur_snapshot_name)
            kvs.update('best_val_preds', kvs['y_pred'])
            kvs.update('best_val_target', kvs['y_true'])

        else:
            if comparator(val_metric, kvs['best_val_metric']):
                print("roc_auc = ", val_metric)
                print(colored('====> ', 'green') + 'Snapshot was saved to', cur_snapshot_name)
                os.remove(kvs['prev_model'])
                torch.save(model.state_dict(), cur_snapshot_name)
                kvs.update('prev_model', cur_snapshot_name)
                kvs.update('best_val_metric', val_metric)
                kvs.update('best_val_preds', kvs['y_pred'])
                kvs.update('best_val_target', kvs['y_true'])

        kvs.save_pkl(os.path.join('sessions/', f'session_fold_[{kvs["cur_fold"]}].pkl'))

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def mapk(self, pred, actual):
        common_values = []
        average_precision = []

        for i in range(len(actual)):
            if pred[i] == actual[i]:
                common_values.append(pred[i])

        return average_precision.append(len(common_values) / len(pred))

    def fit_loader(self,
                   loader,
                   val_loader=None,
                   initial_epoch=0,
                   num_epoch=100,
                   cuda_device=-1,
                   verbose=1):

        kvs = GlobalKVS()
        self.model = self.model.train(mode=True)
        train_aps = torchmetrics.AveragePrecision()  # Binary Classification
        # train_aps = torchmetrics.AveragePrecision(num_classes=4, average='weighted')  # Multi class classification
        # ----------------------------------------------------------------------
        num_inputs = (loader.dataset.num_inputs)
        num_targets = (loader.dataset.num_targets)
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        batch_size = loader.batch_size

        if val_loader is not None:
            num_val_inputs = val_loader.dataset.num_inputs
            num_val_targets = val_loader.dataset.num_targets
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
        has_val_data = val_loader is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._loss_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn,
                                                            self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(self._callbacks+tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': loader.batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            count = 0
            for epoch_idx in range(initial_epoch, num_epoch):
                batches_aps = []
                roc_aucs = []
                kvs.update('cur_epoch', epoch_idx)
                epoch_logs = {}
                callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                batch_logs = {}
                loader_iter = iter(loader)
                for batch_idx in range(num_batches):

                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    input_batch, target_batch = fit_helper.grab_batch_from_loader(loader_iter)

                    if cuda_device >= 0:
                        input_batch, target_batch = fit_helper.move_to_cuda(cuda_device, input_batch, target_batch)
                        input_batch = input_batch.permute(0, 2, 1, 3, 4)  # ch3
                    # ---------------------------------------------
                    if len(target_batch) == 1:
                        print('Singleton.')
                        continue
                    self._optimizer.zero_grad()
                    output_batch = fit_forward_fn(input_batch.float())
                    # if len(target_batch) > 1:
                    #     loss = fit_loss_fn(output_batch, torch.squeeze(target_batch, 1).to(torch.int64))  # multi_class
                    #     batch_aps = train_aps(F.softmax(output_batch), torch.squeeze(target_batch, 1))  # multi_class

                        # target_batch = target_batch.float()  #.float())  # multi_label
                        # loss = fit_loss_fn(F.sigmoid(output_batch), torch.squeeze(target_batch, 2))  # multi_label
                        # batch_aps = train_aps(F.sigmoid(output_batch), torch.squeeze(target_batch, 2))  # multi_label

                    # else:
                    #     loss = fit_loss_fn(output_batch, target_batch)
                    #     batch_aps = train_aps(F.sigmoid(output_batch), target_batch)


                    loss = fit_loss_fn(output_batch.float(), target_batch)  # binary  # .float() for multilabelmulticlass
                    batch_aps = train_aps(F.sigmoid(output_batch), target_batch)  # binary


                    # loss = fit_loss_fn(output_batch, torch.squeeze(target_batch, 1).to(torch.int64))  # multi_class
                    # batch_aps = train_aps(F.softmax(output_batch), torch.squeeze(target_batch.int(), 1))  # multi_class
                    try:
                        tmp = roc_auc_score(y_true=target_batch.detach().cpu().numpy(),
                                            y_score=(output_batch).detach().cpu().numpy())
                        roc_aucs.append(tmp)
                    except:
                        count += 1

                    loss.backward()
                    self._optimizer.step()
                    # ---------------------------------------------

                    if self._has_regularizers:
                        batch_logs['reg_loss'] = self.regularizer_container.current_value
                    if self._has_metrics:
                        metrics_logs = self.metric_container(output_batch, target_batch)
                        batch_logs.update(metrics_logs)

                    batch_logs['loss'] = loss.item()
                    callback_container.on_batch_end(batch_idx, batch_logs)

                if has_val_data:
                    val_epoch_logs, val_epoch_ap, val_epoch_roc_auc = self.evaluate_loader(val_loader,
                                                          cuda_device=cuda_device,
                                                          verbose=verbose)
                    self._in_train_loop = False
                    epoch_logs.update(val_epoch_logs)

                    self.ap.append(train_aps.compute())
                    train_aps.reset()

                    self.eval_ap.append(val_epoch_ap)

                    self.roc_auc.append(np.mean(roc_aucs))
                    self.eval_roc_auc.append(val_epoch_roc_auc)

                    res = dict()

                    if type(val_epoch_ap) == list:
                        res['roc_auc_value'] = val_epoch_roc_auc
                    else:
                        res['roc_auc_value'] = val_epoch_roc_auc.data.tolist()
                    res['val_loss'] = val_epoch_logs['val_loss']
                    res['epoch'] = kvs['cur_epoch']
                    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)
                    self.save_checkpoint(self.model, 'roc_auc_value', 'gt')

                callback_container.on_epoch_end(epoch_idx, epoch_logs)


                # kvs.update('roc_auc', self.roc_auc)
                # kvs.update('eval_roc_auc', self.eval_roc_auc)
                # kvs.update('ap', self.ap)
                # kvs.update('eval_ap', self.eval_ap)
                # kvs.update('acc', self.history['acc'])
                # kvs.update('val_acc', self.history['val_acc'])
                # kvs.update('loss', self.history['loss'])
                # kvs.update('val_loss', self.history['val_loss'])

                if self._stop_training:
                    break
        self.model.train(mode=False)


    def evaluate_loader(self,
                        loader,
                        cuda_device=-1,
                        verbose=1):
        self.model.train(mode=False)
        kvs = GlobalKVS()
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        batch_size = loader.batch_size
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / batch_size))
        # num_batches = 5

        evaluate_helper = _get_helper(self, num_inputs, num_targets)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._loss_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        val_aps = torchmetrics.AveragePrecision()  # Binary Classification
        # val_aps = torchmetrics.AveragePrecision(num_classes=4, average='weighted')  # Multi class classification

        eval_logs = {'val_loss': 0.}
        eval_ap = {'val_ap': 0.}
        eval_roc_auc = {'val_roc_auc': 0.}
        roc_auc = 0
        all_output_batch = []
        all_target_batch = []
        loader_iter = iter(loader)
        count = 0

        if self._has_metrics:
            metric_container = MetricContainer(self._metrics, prefix='val_')
            metric_container.set_helper(evaluate_helper)
            metric_container.reset()

        # samples_seen = 0
        loss = 0
        for batch_idx in range(num_batches):
            input_batch, target_batch = evaluate_helper.grab_batch_from_loader(loader_iter, volatile=True)
            if cuda_device >= 0:
                input_batch, target_batch = evaluate_helper.move_to_cuda(cuda_device, input_batch, target_batch)

            self._optimizer.zero_grad()
            input_batch = input_batch.permute(0, 2, 1, 3, 4)  # 3ch
            output_batch = eval_forward_fn(input_batch)  # .float())
            output_batch = output_batch.detach_().cpu()
            target_batch = target_batch.detach_().cpu()
            # if len(target_batch.size()) > 1:
            #     loss += eval_loss_fn(output_batch, torch.squeeze(target_batch, 1).to(torch.int64)).item()  # multi_class

                # target_batch = target_batch.float()  # .type_as(output_batch)  # multi_label
                # loss += eval_loss_fn(F.sigmoid(output_batch), torch.squeeze(target_batch, 2))  # multi_label
            # else:
            #     loss += eval_loss_fn(output_batch, target_batch).item()
            #     val_batch_aps = val_aps(F.sigmoid(output_batch), target_batch)
            loss += eval_loss_fn(output_batch, target_batch).item()  # binary
            # loss += eval_loss_fn(output_batch, torch.squeeze(target_batch, 1).to(torch.int64)).item()  # multi_class

            # val_batch_aps = val_aps(F.softmax(output_batch), torch.squeeze(target_batch.int(), 1))  # multi_class
            val_batch_aps = val_aps(F.sigmoid(output_batch), target_batch)  # binary
            # val_batch_aps = val_aps(F.sigmoid(output_batch), torch.squeeze(target_batch, 2))  # multi_label

            try:
                tmp = roc_auc_score(y_true=target_batch.detach().cpu().numpy(),
                                    y_score=(output_batch).detach().cpu().numpy())
                roc_auc += tmp
            except:
                count += 1

            all_output_batch.append(output_batch)
            all_target_batch.append(target_batch)

            torch.cuda.empty_cache()
            gc.collect()
            del input_batch
            del output_batch
            del target_batch

        print(count)
        eval_ap['val_ap'] = val_aps.compute()
        eval_logs['val_loss'] = loss/(num_batches - count)
        eval_roc_auc['val_roc_auc'] = roc_auc/(num_batches - count)
        val_aps.reset()

        if self._has_metrics:
            all_preds = th.cat(all_output_batch, dim=0)
            all_targets = th.cat(all_target_batch, dim=0)

            metrics_logs = metric_container(all_preds, all_targets)
            eval_logs.update(metrics_logs)

        self.model.train(mode=True)

        kvs.update('y_true', np.concatenate([item.detach().cpu().numpy() for item in all_targets], axis=0))
        kvs.update('y_pred', np.concatenate([item.detach().cpu().numpy() for item in all_preds], axis=0))
        all_targets.detach()
        all_preds.detach()
        # loss.detach()
        del all_targets
        del all_preds
        # del loss
        torch.cuda.empty_cache()

        return eval_logs, eval_ap['val_ap'], eval_roc_auc['val_roc_auc']



    def compile(self,
                optimizer,
                loss,
                callbacks=None,
                regularizers=None,
                initializers=None,
                constraints=None,
                metrics=None,
                transforms=None):
        self.set_optimizer(optimizer)
        self.set_loss(loss)

        if regularizers is not None:
            self.set_regularizers(regularizers)
            self.regularizer_container = RegularizerContainer(self._regularizers)
            self.regularizer_container.register_forward_hooks(self.model)
        else:
            self._has_regularizers = False

        self.history = History(self)
        self._callbacks = [self.history]
        if callbacks is not None:
            self.set_callbacks(callbacks)


        if initializers is not None:
            self.set_initializers(initializers)
            self.initializer_container = InitializerContainer(self._initializers)
            # actually initialize the model
            self.initializer_container.apply(self.model)

        if constraints is not None:
            self.set_constraints(constraints)
            self.constraint_container = ConstraintContainer(self._constraints)
            self.constraint_container.register_constraints(self.model)
        else:
            self._has_constraints = False

        if metrics is not None:
            self.set_metrics(metrics)
            self.metric_container = MetricContainer(self._metrics)
        else:
            self._has_metrics = False

        if transforms is not None:
            self.set_transforms(transforms)
        else:
            self._has_transforms = False



    def set_metrics(self, metrics):
            metrics = [metrics] if not _is_tuple_or_list(metrics) else metrics
            # metrics = [_validate_metric_input(m) for m in metrics]
            self._has_metrics = True
            self._metrics = metrics

