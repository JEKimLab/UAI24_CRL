import numpy as np
import math
from sklearn import metrics


def _log_value(probs, small_value=1e-30):
    return -np.log(np.maximum(probs, small_value))


def _entr_comp(probs):
    """compute the entropy of the prediction"""
    return np.sum(np.multiply(probs, _log_value(probs)), axis=1)


def _m_entr_comp(probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1 - probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
