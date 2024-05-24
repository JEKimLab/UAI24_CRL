import numpy as np
import sklearn.metrics as metrics


class ThresholdBase(object):
    def __init__(self, shadow_train_scores, shadow_test_scores, target_train_scores, target_test_scores):
        self.s_tr_scores = shadow_train_scores
        self.s_te_scores = shadow_test_scores
        self.t_tr_scores = target_train_scores
        self.t_te_scores = target_test_scores

    def load_labels(self, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes):
        """Load sample labels"""
        self.num_classes = num_classes
        self.s_tr_labels = s_tr_labels
        self.s_te_labels = s_te_labels
        self.t_tr_labels = t_tr_labels
        self.t_te_labels = t_te_labels

    def _thre_setting(self, tr_values, te_values):
        """Select the best threshold"""
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values <= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values > value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre_perclass(self):
        s_tr_values, s_te_values, t_tr_values, t_te_values = \
            self.s_tr_scores, self.s_te_scores, self.t_tr_scores, self.t_te_scores
        """MIA by thresholding per-class feature values """
        t_tr_mem, t_te_non_mem = 0, 0
        thre_list = np.zeros((self.num_classes))
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num], s_te_values[self.s_te_labels == num])
            thre_list[num] = thre
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        info = 'MIA (pre-class threshold): the attack acc is {acc:.3f}'.format(acc=mem_inf_acc)
        print(info)
        return thre_list  # , mem_inf_acc

    def _mem_inf_thre(self, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA by thresholding overall feature values"""
        t_tr_mem, t_te_non_mem = 0, 0
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem += np.sum(t_tr_values >= thre)
        t_te_non_mem += np.sum(t_te_values < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(t_tr_values) + 0.0) + t_te_non_mem / (len(t_te_values) + 0.0))
        info = 'MIA (general threshold): the attack acc is {acc:.3f}'.format(acc=mem_inf_acc)
        print(info)
        return thre  # , mem_inf_acc
