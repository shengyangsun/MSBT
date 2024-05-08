# -*- coding: utf-8 -*-
import logging
import numpy as np
import time
import math

def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    return feat[r:r + t_max]


def uniform_extract(feat, t_max):
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    return feat[r, :]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)


def process_test_feat(feat, length):
    tem_len = len(feat)
    num = math.ceil(tem_len / length)
    if len(feat) < length:
        return pad(feat, length)
    else:
        return pad(feat, num * length)


def Prepare_logger():
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)
    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = 'log/' + date + '.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger