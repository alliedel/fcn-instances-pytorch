import collections
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    try:
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    except:
        import ipdb;
        ipdb.set_trace()
        raise
    return hist


def label_accuracy_score(label_trues, label_preds, n_class=None):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    if n_class is None:
        n_class = label_trues.max() + 1
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    try:
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    except RuntimeWarning:
        import ipdb;
        ipdb.set_trace()
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class TermColors:
    """
    https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def color_text(text, color):
    """
    color can either be a string, like 'OKGREEN', or the value itself, like TermColors.OKGREEN
    """
    color_keys = TermColors.__dict__.keys()
    color_vals = [getattr(TermColors, k) for k in color_keys]
    if color in color_keys:
        color = getattr(TermColors, color)
    elif color in color_vals:
        pass
    else:
        raise Exception('color not recognized: {}\nChoose from: {}, {}'.format(color, color_keys, color_vals))
    return color + text + TermColors.ENDC


def pop_without_del(dictionary, key, default):
    val = dictionary.pop(key, default)
    dictionary[key] = val
    return val


def value_as_string(value):
    if isinstance(value, tuple):
        return ','.join(value_as_string(p) for p in value)
    else:
        return '{}'.format(value)


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


