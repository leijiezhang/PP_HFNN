import torch
from typing import List
from torch.utils.data import Dataset as Dataset_nn
from math_utils import mapminmax


class Dataset(object):
    """
        we suppose the data structure is X: N x D (N is the number of data samples and D is the data sample dimention)
        and the label set Y as: N x 1
    """
    def __init__(self, name, x, y, task):
        """
        init the Dataset class
        :param name: the name of data set
        :param x: features for train
        :param y: labels for regression or classification
        :param task: R for regression C for classification
        """
        self.name = name
        self.X: torch.Tensor = x
        self.Y: torch.Tensor = y
        self.Y_r: torch.Tensor = y
        self.task = task

        # data sequance disorder
        self.shuffle = True

        # partition dataset into several test data and training data
        # centralized partition strategies
        self.partitions = []
        self.__current_partition = 0

    def normalize(self, l_range: torch.int, u_range: int, flag=None):
        self.X = mapminmax(self.X, l_range, u_range)
        if flag is not None:
            d: torch.Tensor = torch.sum(self.X, 1)
            d = d.repeat(1, self.X.shape[1])
            self.X = self.X / d

    def generate_n_partitions(self, n, partition_strategy):

        """
        to run for multiple times, Generate N partitions of the dataset using a given
         partitioning strategy. In semi-supervised mode, two
         strategies must be provided.
        :param n: running times of the dataset, how many times do the experiments operate
        :param partition_strategy:
        :return:
        """

        self.partitions = []
        self.__current_partition = 0

        for i in torch.arange(n):
            current_y = self.Y

            partition_strategy.partition(current_y, int(i), self.shuffle)
            self.partitions.append(partition_strategy)

    def generate_single_partitions(self, partition_strategy):
        self.generate_n_partitions(1, partition_strategy)

    def set_current_partition(self, cur_part):
        self.__current_partition = cur_part
        
    def get_run_set(self, n_fold=None):
        """
        todo:generate training dataset and test dataset by k-folds

        :param n_fold:
        :return: 1st fold datasets for run by default or specified n fold runabel datasets
        """
        x = self.X
        y = self.Y
        partition_strategy = self.partitions[self.__current_partition]
        if n_fold is not None:
            partition_strategy.set_current_folds(n_fold)
        train_idx = partition_strategy.get_train_indexes()
        text_idx = partition_strategy.get_test_indexes()
        train_name = f"{self.name}_train"
        train_data = Dataset(train_name, x[train_idx[0], :], y[train_idx[0], :], self.task)
        test_name = f"{self.name}_test"
        text_data = Dataset(test_name, x[text_idx[0], :], y[text_idx[0], :], self.task)

        return train_data, text_data

    def get_subset_smpl(self, d_partitions):
        """
        todo:divide sample into different several datasets using subsets of original samples
        distribute the dataset to generate n subsets with equal sample numbers
        normally used on training dataset to get sub training dataset
        :param d_partitions: partition strategy
        :return:
        """
        d_partitions.partition(self.Y, 0, self.shuffle)
        d_train_data = []
        x = self.X
        y = self.Y
        for i in torch.arange(d_partitions.get_num_folds()):
            d_partitions.set_current_folds(i)
            disp_part_idx = d_partitions.get_test_indexes()
            disp_part_dataset = Dataset(f"{self.name}_distry", x[disp_part_idx[0], :],
                                        y[disp_part_idx[0], :], self.task)
            d_train_data.append(disp_part_dataset)
        return d_train_data

    def get_subset_fea(self, seperator: List[torch.Tensor]):
        """
        todo:divide feature into different several datasets using sub-features in original feature
        seperate the feature according to the seperator and then form several new datasets
        exm: old feature[1, 2, 3, 4] new feature [1], [2], [3], [4]-->4 datasets with 1 feature in
        each dataset
        :param seperator: the list that includes the index of original features in different sub sets
        :return: sub_dataset_list  the list of sub_datasets
        """
        x_tmp = self.X
        sub_dataset_list: List[Dataset] = []
        for i in torch.arange(len(seperator)):
            x_i = x_tmp[:, seperator[int(i)]]
            name_i = f"{self.name}_sub{i+1}"
            sub_dataset_i = Dataset(name_i, x_i, self.Y, self.task)
            sub_dataset_list.append(sub_dataset_i)

        return sub_dataset_list


class Result(object):
    """
    todo: save results
    """
    def __init__(self, param_mu_list):
        self.loss_c_train = torch.empty(1)
        self.loss_c_train_mean = torch.empty(1)
        self.loss_c_test = torch.empty(1)
        self.loss_c_test_mean = torch.empty(1)
        self.loss_d_train = torch.empty(1)
        self.loss_d_train_mean = torch.empty(1)
        self.loss_d_test = torch.empty(1)
        self.loss_d_test_mean = torch.empty(1)

        self.loss_curve = []
        self.best_idx = 0
        self.param_mu_list = param_mu_list
        self.best_mu = 0.1

        self.loss_c_train_best = torch.empty(1)
        self.loss_c_test_best = torch.empty(1)
        self.loss_d_train_best = torch.empty(1)
        self.loss_d_test_best = torch.empty(1)

    def get_best_idx(self, best_idx):
        self.best_idx = best_idx
        # self.best_mu = self.param_mu_list[best_idx]

        self.loss_c_train_best = self.loss_c_train_mean[best_idx, :]
        self.loss_c_test_best = self.loss_c_test_mean[best_idx, :]
        self.loss_d_train_best = self.loss_d_train_mean[best_idx, :]
        self.loss_d_test_best = self.loss_d_test_mean[best_idx, :]


class DatasetNN(Dataset_nn):
    def __init__(self, x, y=None):
        super(DatasetNN, self).__init__()
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x: torch.Tensor = self.x[index, :]
        y: torch.Tensor = self.y[index, :]
        return x, y


class DatasetH(Dataset_nn):
    """
    this is a pytorch dataset for H list used in FNNdNN model
    """
    def __init__(self, x, y=None):
        super(DatasetH, self).__init__()
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, index):
        x: torch.Tensor = self.x[:, index, :]
        y: torch.Tensor = self.y[index, :]
        return x, y
