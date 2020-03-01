import torch


class FeaSeperator(object):
    """
    in order to divide datasets into several subsets of original samples
    Forms: double level list: List[List[Tensor]]--> first level list denotes the level of fuzzy tree
           and the second level list stands for the real seperators of features of dataset
    """
    def __init__(self, data_name):
        self.data_name = data_name
        self.__seperator = [[]]
        # the structure for rule numbers coresponse to the seperator, each node should have
        # a rule number
        self.__n_rule_tree = [[]]

    def set_seperator_by_slice_window(self, window_size, step=1, n_level=2):
        """set seperator using slice window"""
        n_fea = 18
        if self.data_name.find('HRSS') != -1:
            n_fea = 18
        elif self.data_name.find('seed_c62') != -1:
            n_fea = 31
        elif self.data_name.find('seed_c12') != -1:
            n_fea = 60
        elif self.data_name.find('seed_c9') != -1:
            n_fea = 45
        elif self.data_name.find('seed_c6') != -1:
            n_fea = 30
        elif self.data_name.find('seed_c4') != -1:
            n_fea = 20
        elif self.data_name.find('eegDual_subj') != -1:
            n_fea = 24
        elif self.data_name.find('eegDual_subj') != -1:
            n_fea = 16

        seperator = slide_window(n_fea, window_size, step, n_level)
        self.__seperator = seperator

    def set_seperator_by_stride_window(self, stride_len, n_level=2):
        """set seperator using slice window"""
        n_fea = 18
        if self.data_name.find('HRSS') != -1:
            n_fea = 18
        elif self.data_name.find('seed_c62') != -1:
            n_fea = 31
        elif self.data_name.find('seed_c12') != -1:
            n_fea = 60
        elif self.data_name.find('seed_c9') != -1:
            n_fea = 45
        elif self.data_name.find('seed_c6') != -1:
            n_fea = 30
        elif self.data_name.find('seed_c4') != -1:
            n_fea = 20
        elif self.data_name.find('eegDual_subj') != -1:
            n_fea = 24
        elif self.data_name.find('eegDual_subj') != -1:
            n_fea = 16

        seperator = stride_window(n_fea, stride_len, n_level)
        self.__seperator = seperator

    def set_seperator_by_random_pick(self, window_size, n_repeat=2, n_level=2):
        """set seperator using slice window"""
        n_fea = 18
        if self.data_name.find('HRSS') != -1:
            n_fea = 18
        elif self.data_name.find('seed_c62') != -1:
            n_fea = 31
        elif self.data_name.find('seed_c12') != -1:
            n_fea = 60
        elif self.data_name.find('seed_c9') != -1:
            n_fea = 45
        elif self.data_name.find('seed_c6') != -1:
            n_fea = 30
        elif self.data_name.find('seed_c4') != -1:
            n_fea = 20
        elif self.data_name.find('eegDual_subj') != -1:
            n_fea = 24
        elif self.data_name.find('eegDual_subj') != -1:
            n_fea = 16

        seperator = random_pick(n_fea, window_size, n_repeat, n_level)

        self.__seperator = seperator

    def set_seperator_by_no_seperate(self):
        """do not seperate features"""
        seperator = [[]]
        self.__seperator = seperator

    def generate_n_rule_tree(self, n_rule_general):
        """
        generate tree of rule numbers according to the seperator structure
        :param n_rule_general: general rule number for all fnn neurons
        :return:
        """
        n_rule_tree = []
        seperator = self.get_seperator()
        for i in torch.arange(len(seperator)):
            len_seperator_sub = len(seperator[int(i)])
            if len_seperator_sub == 0:
                len_seperator_sub = 1
            n_rule_brunch = n_rule_general * torch.ones(len_seperator_sub)
            n_rule_tree.append(n_rule_brunch)
        self.__n_rule_tree = n_rule_tree

    def set_n_rule_tree(self, row_idx, column_idx, n_rule):
        """
        set n_rule in n_rule_tree by specify row index and column index velue
        (1, 1) is the start position, o denotes the whole index or column
        :param row_idx:
        :param column_idx:
        :param n_rule:
        :return:
        """
        n_rule_tree = self.get_n_rule_tree()
        if row_idx == 0:
            for i in torch.arange(len(n_rule_tree)):
                n_rule_brunch = n_rule * torch.ones(len(n_rule_tree[int(i)]))
                n_rule_tree[int(i)] = n_rule_brunch
        elif column_idx == 0:
            n_rule_brunch = n_rule * torch.ones(len(n_rule_tree[int(row_idx - 1)]))
            n_rule_tree[int(row_idx - 1)] = n_rule_brunch
        else:
            n_rule_tree[int(row_idx - 1)][int(column_idx - 1)] = n_rule
        self.__n_rule_tree = n_rule_tree

    def get_n_rule_tree(self):
        return self.__n_rule_tree

    def get_seperator(self):
        return self.__seperator


def slide_window(n_fea, window_size, step=1, n_level=2):
    """
    slide window to get the index of feature seperators
    :param n_fea:
    :param window_size:
    :param step:
    :param n_level:
    :return:
    """
    n_fea_tmp = n_fea
    level_idx = 1
    seperator = []

    while True:
        idx_sub = 0
        seperator_sub = []
        if not window_size < n_fea_tmp or not level_idx < n_level:
            seperator.append([])
            break
        while idx_sub + window_size <= n_fea_tmp:
            fea_idx = torch.linspace(idx_sub, idx_sub + window_size - 1, window_size)
            fea_idx_real = fea_idx % n_fea_tmp
            fea_idx_real = fea_idx_real.long()
            seperator_sub.append(fea_idx_real)
            idx_sub = idx_sub + step
        n_fea_tmp = len(seperator_sub)
        seperator.append(seperator_sub)
        level_idx = level_idx + 1
    return seperator


def stride_window(n_fea, stride_len, n_level=2):
    """
    using stride instead of slide window to get the index of feature seperators
    :param n_fea:
    :param stride_len:
    :param n_level:
    :return:
    """
    n_fea_tmp = n_fea
    level_idx = 1
    seperator = []

    while True:
        idx_sub = 0
        seperator_sub = []
        if not stride_len < n_fea_tmp or not level_idx < n_level:
            seperator.append([])
            break
        while idx_sub < stride_len:
            fea_idx = torch.arange(idx_sub, n_fea, stride_len)
            fea_idx_real = fea_idx.long()
            seperator_sub.append(fea_idx_real)
            idx_sub = idx_sub + 1
        n_fea_tmp = len(seperator_sub)
        seperator.append(seperator_sub)
        level_idx = level_idx + 1
    return seperator


def random_pick(n_fea, window_size, n_repeat=3, n_level=2):
    """
    randomly picking to get the index of feature seperators
    :param n_fea:
    :param window_size:
    :param n_repeat: the times of repeat selecting
    :param n_level:
    :return:
    """
    n_fea_tmp = n_fea
    step = window_size
    level_idx = 1
    seperator = []

    while True:
        seperator_sub = []
        if not window_size < n_fea_tmp or not level_idx < n_level:
            seperator.append([])
            break

        for i in torch.arange(n_repeat):
            idx_sub = 0
            # disoder the sequnce of features
            fea_seq = torch.randperm(n_fea_tmp)
            while idx_sub + window_size < n_fea_tmp:
                fea_idx = torch.linspace(idx_sub, idx_sub + window_size - 1, window_size).long()
                fea_idx_real = fea_seq[fea_idx] % n_fea_tmp
                fea_idx_real = fea_idx_real.long()
                seperator_sub.append(fea_idx_real)
                idx_sub = idx_sub + step

        n_fea_tmp = len(seperator_sub)
        seperator.append(seperator_sub)
        level_idx = level_idx + 1
    return seperator
