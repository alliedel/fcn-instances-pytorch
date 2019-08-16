import argparse
import collections
import logging
import os

import numpy as np
import torch


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


def color_text(text, color=None):
    """
    color can either be a string, like 'OKGREEN', or the value itself, like TermColors.OKGREEN
    """
    if color is None:
        return text
    color_keys = TermColors.__dict__.keys()
    color_vals = [getattr(TermColors, k) for k in color_keys]
    if color in color_keys:
        color = getattr(TermColors, color)
    elif color in color_vals:
        pass
    else:
        raise Exception('color not recognized: {}\nChoose from: {}, {}'.format(color, color_keys, color_vals))
    return color + text + TermColors.ENDC


class AttrDict(object):
    def __init__(self, init=None):
        if init is not None:
            self.__dict__.update(init)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)


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


def mkdir_if_needed(dirname):
    if os.path.isfile(dirname):
        raise ValueError('{} is an existing file!  Cannot make directory.'.format(dirname))
    if not os.path.isdir(dirname):
        os.mkdir(dirname)


def get_logger(name='my_logger', file=None):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        # logger.setLevel(logging.DEBUG)

        if file == None:
            file = '/tmp/my_log.log'
            print('Logging file in {}'.format(file))

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def unique(x):
    if type(x) == np.ndarray:
        return np.unique(x)
    elif type(x) == list:
        return list(set(list(x)))
    else:
        raise TypeError('unique not implemented for type {}'.format(type(x)))


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_array_size(obj):
    if type(obj) is np.ndarray:
        return obj.shape
    elif torch.is_tensor(obj):
        return obj.size()
    else:
        return None


def pairwise_and(list1, list2):
    return [a and b for a, b in zip(list1, list2)]


def pairwise_or(list1, list2):
    return [a or b for a, b in zip(list1, list2)]


def y_or_n_input(msg_to_user, allowed_chars=('y', 'n'), default=None, convert_to_bool_is_y=False):
    empty_response_char = ''
    y_or_n = input(msg_to_user + ' ')
    if default is not None:
        allowed_chars = allowed_chars if '' in allowed_chars else list(allowed_chars) + [empty_response_char]
    while y_or_n not in allowed_chars:
        print('Answer with one of {}. '.format(allowed_chars))
        y_or_n = input(msg_to_user)
    if y_or_n is empty_response_char and default is not None:
        y_or_n = default
    if convert_to_bool_is_y:
        return y_or_n == 'y' or y_or_n == 'Y'
    return y_or_n


import os, tempfile


def symlink(target, link_name, overwrite=False):
    '''
    Source: https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python/55742015#55742015
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    '''

    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if os.path.isdir(link_name):
            raise IsADirectoryError(f"Cannot symlink over existing directory: '{link_name}'")
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


def rgb2hex(r, g, b):
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
    return '#%02x%02x%02x' % (r, g, b)
