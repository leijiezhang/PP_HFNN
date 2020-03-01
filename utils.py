import torch
import scipy.io as sio
from dataset import Dataset
import logging
import os
import time
import csv


def load_data(dataset_str, sub_fold):
    dir_dataset = f"./datasets/{sub_fold}/{dataset_str}.pt"

    load_data = torch.load(dir_dataset)
    dataset_name = load_data['name']
    x: torch.Tensor = load_data['X']
    y: torch.Tensor = load_data['Y']

    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    task = load_data['task']
    dataset = Dataset(dataset_name, x, y, task)

    if 'Y_r' in load_data:
        y_r: torch.Tensor = load_data['Y_r']
        dataset.Y_r = y_r

    if 'seperator' in load_data:
        seperator = load_data['seperator']
        dataset.seperator = seperator

    # dataset.normalize(-1, 1)
    return dataset


def dataset_parse(dataset_name, sub_fold):
    """
    todo: parse dataset from .mat to .pt
    :param dataset_name:
    :param sub_fold:
    :return:
    """
    dir_dataset = f"./datasets/{sub_fold}/{dataset_name}.mat"
    load_data = sio.loadmat(dir_dataset)
    dataset_name = load_data['name'][0]
    x_orig = load_data['X']
    y_orig = load_data['Y']
    x = torch.tensor(x_orig).double()
    # y_tmp = []
    # for i in torch.arange(len(y_orig)):
    #     y_tmp.append(float(y_orig[i, :]))
    y = torch.tensor(y_orig).double()
    task = load_data['task'][0]
    data_save = dict()
    data_save['task'] = task
    data_save['name'] = dataset_name
    data_save['X'] = x
    if task == 'C':
        y_min = torch.min(y)
        y_gap = y_min - 0
        y = y - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['Y_r'] = y_c
        data_save['Y'] = y
    else:
        data_save['Y'] = y

    dir_dataset = f"./datasets/{sub_fold}/{dataset_name}.pt"
    torch.save(data_save, dir_dataset)


def eeg_dataset_parse(dataset_name, sub_fold):
    """
    todo: parse dataset from .mat to .pt
    :param dataset_name:
    :param sub_fold:
    :return:
    """
    dir_dataset = f"./datasets/{sub_fold}/{dataset_name}.mat"
    load_data = sio.loadmat(dir_dataset)
    dataset_name = load_data['name'][0]
    x_train_orig = load_data['X_train']
    y_train_orig = load_data['Y_train']
    x_test_orig = load_data['X_test']
    y_test_orig = load_data['Y_test']
    x_train = torch.tensor(x_train_orig).double()
    x_test = torch.tensor(x_test_orig).double()

    y_tmp = []
    for i in torch.arange(len(y_train_orig)):
        y_tmp.append(float(y_train_orig[i]))
    y_train = torch.tensor(y_tmp).double()
    y_tmp = []
    for i in torch.arange(len(y_test_orig)):
        y_tmp.append(float(y_test_orig[i]))
    y_test = torch.tensor(y_tmp).double()

    task = load_data['task'][0]
    data_save = dict()
    data_save['task'] = task
    data_save['name'] = dataset_name

    data_save['x_train'] = x_train
    data_save['x_test'] = x_test

    if task == 'C':
        y_min = torch.min(y_test)
        y_gap = y_min - 0
        y = y_test - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['y_test_r'] = y_c
        data_save['y_test'] = y
    else:
        data_save['y_test'] = y_test.unsqueeze(1)

    if task == 'C':
        y_min = torch.min(y_train)
        y_gap = y_min - 0
        y = y_train - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['y_train_r'] = y_c
        data_save['y_train'] = y
    else:
        data_save['y_train'] = y_train.unsqueeze(1)

    dir_dataset = f"./datasets/{sub_fold}/{dataset_name}.pt"
    torch.save(data_save, dir_dataset)


def seed_dataset_parse(n_channel, idx_exp, idx_subj):
    """
    todo: parse dataset from .mat to .pt
    :param idx_exp:
    :param n_channel: channel number
    :param idx_subj: experiment indices
    :return:
    """
    dataset_name = f"seed_subj{idx_subj}"
    dir_dataset = f"./datasets/seed/channel{n_channel}/experiment{idx_exp}/{dataset_name}.mat"
    dataset_name = f"seed_c{n_channel}_e{idx_exp}_subj{idx_subj}"

    load_data = sio.loadmat(dir_dataset)
    x_train_orig = load_data['X_train']
    y_train_orig = load_data['Y_train']
    x_test_orig = load_data['X_test']
    y_test_orig = load_data['Y_test']
    x_train = torch.tensor(x_train_orig).double()
    x_test = torch.tensor(x_test_orig).double()

    y_tmp = []
    for i in torch.arange(len(y_train_orig)):
        y_tmp.append(float(y_train_orig[i]))
    y_train = torch.tensor(y_tmp).double()
    y_tmp = []
    for i in torch.arange(len(y_test_orig)):
        y_tmp.append(float(y_test_orig[i]))
    y_test = torch.tensor(y_tmp).double()

    task = load_data['task'][0]
    data_save = dict()
    data_save['task'] = task
    data_save['name'] = dataset_name

    data_save['x_train'] = x_train
    data_save['x_test'] = x_test

    if task == 'C':
        y_min = torch.min(y_test)
        y_gap = y_min - 0
        y = y_test - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['y_test_r'] = y_c
        data_save['y_test'] = y
    else:
        data_save['y_test'] = y_test.unsqueeze(1)

    if task == 'C':
        y_min = torch.min(y_train)
        y_gap = y_min - 0
        y = y_train - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['y_train_r'] = y_c
        data_save['y_train'] = y
    else:
        data_save['y_train'] = y_train.unsqueeze(1)

    dir_dataset = f"./datasets/seed/channel{n_channel}/{dataset_name}.pt"
    torch.save(data_save, dir_dataset)


def dataset_from_kaggle(dataset_name='heart', label_idx=14, task = 'C'):
    """
    parse data from kaggle
    :param dataset_name:
    :param label_idx:
    :return:
    """
    label_idx = label_idx - 1

    dir_dataset = f"./datasets/{dataset_name}.csv"
    with open(dir_dataset, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        n_fea = len(fieldnames)
        data = torch.empty(0, n_fea)
        for row in reader:
            row = list(map(float, row))
            data_row = torch.tensor(row).unsqueeze(0)
            data = torch.cat((data, data_row), 0)
    y = data[:, label_idx]
    x = data[:, torch.arange(data.size(1)) != label_idx].double()
    data_save = dict()
    data_save['task'] = task
    data_save['name'] = dataset_name
    data_save['X'] = x
    if task == 'C':
        y_min = torch.min(y)
        y_gap = y_min - 0
        y = y - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['Y_r'] = y
        data_save['Y'] = y_c
    else:
        data_save['Y'] = y

    dir_dataset = f"./datasets/{dataset_name}.pt"
    torch.save(data_save, dir_dataset)


class Logger(object):
    def __init__(self, to_file=False, clevel=logging.DEBUG, Flevel=logging.DEBUG):
        self.to_file = to_file
        # create dictionary
        file_name = f"./log/log_{time.strftime('%H-%M-%S ',time.localtime(time.time()))}.log"
        self.file_name = file_name
        if not os.path.exists(file_name):
            folder_name = './log'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # set CMD dairy
        self.sh = logging.StreamHandler()
        if self.to_file:
            # set log file
            self.fh = logging.FileHandler(file_name, encoding='utf-8')

    def debug(self, message):
        self.set_color('\033[0;34m%s\033[0m', logging.DEBUG)
        self.logger.debug(message)

    def info(self, message):
        self.set_color('\033[0;30m%s\033[0m', logging.INFO)
        self.logger.info(message)

    def war(self, message):
        self.set_color('\033[0;32m%s\033[0m', logging.WARNING)
        self.logger.warning(message)

    def error(self, message):
        self.set_color('\033[0;31m%s\033[0m', logging.ERROR)
        self.logger.error(message)

    def cri(self, message):
        self.set_color('\033[0;35m%s\033[0m', logging.CRITICAL)
        self.logger.critical(message)

    def set_color(self, color, level):
        fmt = logging.Formatter(color % '[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set CMD dairy
        self.sh.setFormatter(fmt)
        self.sh.setLevel(level)
        self.logger.addHandler(self.sh)
        if self.to_file:
            # set log file
            self.fh.setFormatter(fmt)
            self.fh.setLevel(level)
            self.logger.addHandler(self.fh)

