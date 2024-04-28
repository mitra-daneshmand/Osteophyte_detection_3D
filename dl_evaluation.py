import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F
import datasets
import models
from Bootstraping_curves import curves
from sklearn.metrics import roc_curve
from kvs import GlobalKVS
from args import parse_args

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models import BalancedAccuracy
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import sem, t

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import sem, t


def balanced_accuracy_with_ci(y_true, y_pred, confidence_level=0.95):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    n = len(y_true)
    std_err = sem([1 if p == t else 0 for p, t in zip(y_pred, y_true)])
    h = std_err * t.ppf((1 + confidence_level) / 2, n - 1)
    conf_int = (balanced_acc - h, balanced_acc + h)

    return balanced_acc, conf_int


kvs = GlobalKVS()
args = parse_args()
save_dir = 'sessions_final/{}/{}/'.format(args.tissue, args.target_comp)
# save_dir = 'sessions'
kvs.update('save_dir', save_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
test_res = 0
read_test_set = 0
'''
for fold in range(5):
    warnings.filterwarnings('ignore')
    with open((os.path.join(save_dir, f'session_fold_[{fold}].pkl')), 'rb') as f:
        session_snapshot = pickle.load(f)
    if read_test_set == 0:
        test_set = session_snapshot['test_set'][0]

        ################### KL based evaluation ###################
        inds = test_set[(test_set['KL'] > 1)].index  # no OA
        # inds = test_set[(test_set['KL'] != 2)].index  # early OA
        # inds = test_set[(test_set['KL'] < 3)].index  # severe OA
        test_set.drop(inds, inplace=True)
        test_set.reset_index(inplace=True, drop=True)

        mean = session_snapshot['mean'][0]
        std = session_snapshot['std'][0]

        read_test_set = 1

    net, trainer, cuda_device = models.build_model(args.network, args.model_depth, args.learning_rate,
                                                   args.weight_decay)
    clf_prog = net.to(device)
    model_dirs = glob.glob(os.path.join(save_dir, 'fold_{}_epoch_'.format(fold) + '*'))
    clf_prog.load_state_dict(torch.load(model_dirs[0], map_location='cuda:0'))
    clf_prog.eval()

    test_dataset = datasets.KneeDataset(test_set, np.array(test_set['Target']))

    loader = DataLoader(test_dataset,
                        batch_size=16,
                        sampler=SequentialSampler(test_dataset),
                        num_workers=16)
    test_dataset.mean, test_dataset.std = mean, std
    kvs.update('mean', mean)
    kvs.update('std', std)

    preds_prog_fold = []
    y = []
    for batch_id, sample in enumerate(
            tqdm(loader, total=len(loader), desc='Prediction from fold {}'.format(fold))):
        local_batch, local_labels = sample
        y.append(local_labels)

        test_sample = sample[0].permute(0, 2, 1, 3, 4)  # 3ch
        # test_sample = sample[0]  # 1ch
        test_sample = test_sample.to(device)

        probs_prog = clf_prog(test_sample.float())
        preds_prog_fold.append(F.sigmoid(probs_prog).cpu().detach().numpy())

        torch.cuda.empty_cache()
        test_sample.detach()
        probs_prog.detach()
        del test_sample
        del probs_prog

    y_true = np.vstack([x.numpy() for x in y])
    preds_prog_fold = np.vstack(preds_prog_fold)
    test_res += preds_prog_fold

test_res /= 5

kvs.update('y_pred', test_res)
kvs.update('y_true', y_true)
kvs.update('eval_type', 'test')
np.savez_compressed(os.path.join(save_dir, 'results.npz'), y_true=y_true, y_pred=test_res)

curves()
'''

result = np.load(os.path.join(save_dir, 'results.npz'))
fold = 0
with open((os.path.join(save_dir, f'session_fold_[{fold}].pkl')), 'rb') as f:
    session_snapshot = pickle.load(f)
    if read_test_set == 0:
        test_set = session_snapshot['test_set'][0]

        ################### KL based evaluation ###################
        # test_set.reset_index(inplace=True, drop=True)
        # inds = test_set[(test_set['KL'] == 1) | (test_set['KL'] == 0)].index  # no OA
        # inds = test_set[(test_set['KL'] == 2)].index  # early OA
        # inds = test_set[(test_set['KL'] == 3) | (test_set['KL'] == 4)].index  # severe OA
        # test_set.drop(inds, inplace=True)
        # test_set = test_set.iloc[inds]


y_true = result['y_true']#[inds]
test_res = result['y_pred']#[inds]

# kappas = cohen_kappa_score(y_true, test_res.round())
# print('Cohen kappa is: ', kappas)
#
# mse = mean_squared_error(y_true, test_res)
# print('MSE is: ', mse)



df = pd.DataFrame(columns=['ID', 'SIDE', 'OSTs', 'preds', 'probs'])
df['ID'] = test_set['ID']
df['SIDE'] = test_set['SIDE']
df['OSTs'] = test_set[session_snapshot['args'][0].target_comp]
df['probs'] = test_res

# test_res = test_res.round()  # Binary
# test_res = np.argmax(test_res, axis=1)
# test_res = test_res.reshape(test_res.shape[0], 1)

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# precision, recall, thresholds = precision_recall_curve(y_true, test_res)
# scores = (2 * precision * recall) / (precision + recall)

fpr, tpr, thresholds = roc_curve(y_true, test_res)
scores = tpr - fpr

ix = np.argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

test_res = (test_res > thresholds[ix]).astype('int')
df['preds'] = test_res


####################BA#########################
confidence_level = 0.95
balanced_acc, conf_int = balanced_accuracy_with_ci(y_true, test_res, confidence_level=0.95)

# Print the results
print("Balanced accuracy: {:.2f}".format(balanced_acc))
print("Confidence interval ({:.0%}): [{:.2f}, {:.2f}, {}]".format(confidence_level, conf_int[0], conf_int[1], (conf_int[1]-conf_int[0])/2))





bal_acc = BalancedAccuracy()
bal_acc.update(test_res, y_true)
acc = bal_acc.compute()
print('Balanced Accuracy is: ', acc)

cm = pd.DataFrame(columns=[0, 1])

# for i in range(4):
#     indx = df[(df['OSTs'] == i) & (df['preds'] == int(i == 0))].index
#     cm.loc[i, int(i != 0)] = len(indx)
#     indx = df[(df['OSTs'] == i) & (df['preds'] != int(i == 0))].index
#     cm.loc[i, int(i == 0)] = len(indx)

indx = df[(df['OSTs'] == 0) & (df['preds'] == 0)].index
cm.loc[0, 0] = len(indx)
indx = df[(df['OSTs'] == 0) & (df['preds'] != 0)].index
cm.loc[0, 1] = len(indx)

indx = df[(df['OSTs'] == 1) & (df['preds'] == 1)].index
cm.loc[1, 1] = len(indx)
indx = df[(df['OSTs'] == 1) & (df['preds'] != 1)].index
cm.loc[1, 0] = len(indx)

indx = df[(df['OSTs'] == 2) & (df['preds'] == 1)].index
cm.loc[2, 1] = len(indx)
indx = df[(df['OSTs'] == 2) & (df['preds'] != 1)].index
cm.loc[2, 0] = len(indx)

indx = df[(df['OSTs'] == 3) & (df['preds'] == 1)].index
cm.loc[3, 1] = len(indx)
indx = df[(df['OSTs'] == 3) & (df['preds'] != 1)].index
cm.loc[3, 0] = len(indx)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')


table = ax.table(cellText=cm.values, colWidths=[0.25] * len(cm.columns),
                 rowLabels=cm.index,
                 colLabels=cm.columns,
                 cellLoc='center', rowLoc='center',
                 loc='center')
fig.tight_layout()


cm_total = confusion_matrix(y_true, test_res)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
disp.plot(cmap="Blues", values_format='d')

plt.show()
# fig.savefig(os.path.join(save_dir, 'conf_class_based.pdf'), bbox_inches='tight')

sen_total = cm_total[1, 1] / (cm_total[1, 1] + cm_total[1, 0])
spec_total = cm_total[0, 0] / (cm_total[0, 0] + cm_total[0, 1])

sen_1 = cm.loc[1, 1] / (cm.loc[1, 1] + cm.loc[1, 0])
sen_2 = cm.loc[2, 1] / (cm.loc[2, 1] + cm.loc[2, 0])
sen_3 = cm.loc[3, 1] / (cm.loc[3, 1] + cm.loc[3, 0])

print(args.target_comp)
print('Total sensitivity = ', sen_total)
print('Total specificity = ', spec_total)
print('######################################')
print('Class1 sensitivity = ', sen_1)
print('Class2 sensitivity = ', sen_2)
print('Class3 sensitivity = ', sen_3)
