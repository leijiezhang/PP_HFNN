import abc
import torch
from sklearn.model_selection import KFold


class PartitionStrategy(object):
    """
    the base class of partition
    """
    def __init__(self):
        self.current_fold = 0
        self.num_folds = 0

    @abc.abstractmethod
    def partition(self, y: torch.Tensor, random_state, is_shuffle):
        return self

    @abc.abstractmethod
    def get_train_indexes(self):
        train_idx = []
        num_train_idx = []
        return train_idx, num_train_idx

    @abc.abstractmethod
    def get_test_indexes(self):
        test_idx = []
        num_test_idx = []
        return test_idx, num_test_idx

    @abc.abstractmethod
    def get_description(self):
        d = []
        return d

    def get_num_folds(self):
        return self.num_folds

    def set_current_folds(self, cur_fold):
        self.current_fold = cur_fold


class KFoldPartition(PartitionStrategy):
    def __init__(self, k):
        super(KFoldPartition, self).__init__()
        self.K = k
        self.train_indexes = []
        self.test_indexes = []
        self.num_folds = k
        self.y_length = 0

    def partition(self, y: torch.Tensor, random_state=0, is_shuffle=True):
        kf = KFold(n_splits=self.K, random_state=random_state, shuffle=is_shuffle)
        self.train_indexes = []
        self.test_indexes = []
        for train_index, test_index in kf.split(y):
            self.train_indexes.append(train_index)
            self.test_indexes.append(test_index)
        self.y_length = y.shape[0]
        self.set_current_folds(0)

    def get_train_indexes(self):
        train_idx = self.train_indexes[self.current_fold]
        num_train_idx = train_idx.shape[0]
        return train_idx, num_train_idx

    def get_test_indexes(self):
        test_idx = self.test_indexes[self.current_fold]
        num_test_idx = test_idx.shape[0]
        return test_idx, num_test_idx

    def get_description(self):
        d = ('%i-fold partition',self.K)
        return d


# class ExactPartition(PartitionStrategy):
#     def __init__(self, n_train_smpl, n_test_smpl):
#         super(ExactPartition, self).__init__()
#         self.n_train_smpl = n_train_smpl
#         self.n_test_smpl = n_test_smpl
#         self.num_folds = 1
#
#     def partition(self, y: torch.Tensor):
#         if y.shape[0] < self.n_train_smpl + self.n_test_smpl:
#             raise Exception('One of the datasets has not enough samples for the exact partitioning')
#         elif y.shape[0] > self.n_train_smpl + self.n_test_smpl:
#             raise Exception('One of the datasets has exeed samples for the exact partitioning')
#         else:
#             pass
#
#     def get_train_indexes(self):
#         train_idx = self.train_indexes[self.current_fold]
#         num_train_idx = train_idx.shape[0]
#         return train_idx, num_train_idx
#
#     def get_test_indexes(self):
#         test_idx = self.test_indexes[self.current_fold]
#         num_test_idx = test_idx.shape[0]
#         return test_idx, num_test_idx
#
#     def get_description(self):
#         d = ('%i-fold partition',self.K)
#         return d
