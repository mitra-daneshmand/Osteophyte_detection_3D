import os.path

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from kvs import GlobalKVS


class curves():

    def __init__(self, sub_features=None, n_bootstrap=2000, seed=12345):
        self.sub_features = sub_features
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self._init_loader()

    def _init_loader(self):
        kvs = GlobalKVS()

        lower_auc, upper_auc, auc_v, fprs, tprs = self.roc_bootstrap(kvs['y_true'] > 0, kvs['y_pred'])
        lower_pr, upper_pr, ap_v, precisions, recalls = self.ap_bootstrap(kvs['y_true'] > 0, kvs['y_pred'])

        plt.clf()
        plt.subplots(figsize=(20, 20))
        ####################ROCCurves#######################
        plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='No Skill')

        plt.plot(fprs, tprs, label="{} (AUC={:.2f}[{:.2f}-{:.2f}])".format(kvs['eval_type'], auc_v, lower_auc, upper_auc))

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("Flase Positive Rate", fontsize=20)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=20)

        plt.title('ROC Curve Analysis', fontweight='bold', fontsize=20)
        plt.legend(prop={'size': 20}, loc='lower center', ncol=1)

        plt.savefig(os.path.join(kvs['save_dir'], '{}_ROC.pdf'.format(kvs['eval_type'])), bbox_inches='tight')
        plt.show()
        plt.subplots(figsize=(20, 20))
        #####################PRCurves#######################
        no_skill = kvs['y_true'].sum() / len(kvs['y_true'])
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

        plt.plot(recalls, precisions, label="{} (AP={:.2f}[{:.2f}-{:.2f}])".format(kvs['eval_type'], ap_v, lower_pr, upper_pr))

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("Recall", fontsize=20)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("Precision", fontsize=20)

        plt.title('PR Curve Analysis', fontweight='bold', fontsize=20)
        plt.legend(prop={'size': 20}, loc='lower center', ncol=1)

        plt.savefig(os.path.join(kvs['save_dir'], '{}_PR.pdf'.format(kvs['eval_type'])), bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

    def roc_bootstrap(self, test_y, preds):
        print('len test_y', len(test_y))
        print('len preds', len(preds))
        init_auc = roc_auc_score(test_y, preds)
        print('No bootstrapping: auc = {0}'.format(init_auc))
        np.random.seed(self.seed)
        aucs = np.array([], dtype=np.int64).reshape(0, 1)
        k = 0
        for _ in tqdm(range(self.n_bootstrap), total=self.n_bootstrap, desc='Bootstrap'):
            ind = np.random.choice(test_y.shape[0], test_y.shape[0])

            if test_y[ind].sum() == 0:
                continue
            try:
                aucs = np.vstack((aucs, np.array(roc_auc_score(test_y[ind], preds[ind]))))
            except ValueError:
                k += 1
                continue

        if k > 0:
            print('{0} exceptions occurred. Check grade distribution'.format(k))
        auc_v = np.mean(aucs)
        print('Bootstrapping: auc = {0}'.format(auc_v))

        lower_auc, upper_auc = np.percentile(aucs, 1.96), np.percentile(aucs, 95)

        print('AUC:', np.round(auc_v, 5))
        print(f'CI [{lower_auc:.5f}, {upper_auc:.5f}]')

        fpr, tpr, _ = roc_curve(test_y, preds, drop_intermediate=False)

        return lower_auc, upper_auc, auc_v, fpr, tpr

    def ap_bootstrap(self, test_y, preds):
        init_ap = average_precision_score(test_y.flatten(), preds.flatten())
        print('No bootstrapping: AP = {0}'.format(init_ap))
        np.random.seed(self.seed)
        aps = np.array([], dtype=np.int64).reshape(0, 1)
        k = 0
        for _ in tqdm(range(self.n_bootstrap), total=self.n_bootstrap, desc='Bootstrap'):
            ind = np.random.choice(test_y.shape[0], test_y.shape[0])

            if test_y[ind].sum() == 0:
                continue
            try:
                aps = np.vstack((aps, np.array(
                    average_precision_score(np.array(test_y).flatten()[ind], np.array(preds).flatten()[ind]))))
            except ValueError:
                k += 1
                continue

        if k > 0:
            print('{0} exceptions occurred. Check grade distribution'.format(k))
        ap_v = np.mean(aps)
        print('Bootstrapping: ap = {0}'.format(ap_v))

        lower_pr, upper_pr = np.percentile(aps, 1.96), np.percentile(aps, 95)

        print('AP:', np.round(ap_v, 5))
        print(f'CI [{lower_pr:.5f}, {upper_pr:.5f}]')
        precisions, recalls, _ = precision_recall_curve(test_y, preds)
        return lower_pr, upper_pr, ap_v, precisions, recalls
