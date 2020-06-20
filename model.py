from neuron import Neuron
from dataset import Dataset, DatasetH
from seperator import FeaSeperator
from typing import List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from math_utils import cal_fc_w
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import abc
import os


class NetBase(object):
    def __init__(self, neuron_seed: Neuron):
        self.__neuron_seed: Neuron = neuron_seed

    def get_neuron_seed(self):
        return self.__neuron_seed

    def set_neuron_seed(self, neuron_seed: Neuron):
        self.__neuron_seed = neuron_seed

    @abc.abstractmethod
    def forward(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        pass

    @abc.abstractmethod
    def clear(self):
        pass


class TreeNet(NetBase):
    def __init__(self, neuron_seed: Neuron):
        super(TreeNet, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator: FeaSeperator = kwargs['seperator']
        fea_seperator = seperator.get_seperator()
        n_rules_tree = seperator.get_n_rule_tree()

        data_tmp = data
        neuron_seed = self.get_neuron_seed()

        rules_tree: List[List[type(neuron_seed)]] = []
        for j in torch.arange(len(fea_seperator)):
            sub_seperator = fea_seperator[int(j)]
            if len(sub_seperator):
                sub_dataset_list = data_tmp.get_subset_fea(sub_seperator)
                # set level j
                rules_sub: List[type(neuron_seed)] = []
                y_sub = torch.empty(data.Y.shape[0], 0).double()
                for i in torch.arange(len(sub_seperator)):
                    neuron_seed.clear()
                    neuron_c = neuron_seed.clone()
                    sub_dataset_i = sub_dataset_list[i]
                    kwargs['data'] = sub_dataset_i
                    kwargs['n_rules'] = int(n_rules_tree[j][i])
                    neuron_c.forward(**kwargs)
                    y_i = neuron_c.predict(sub_dataset_i)
                    rules_sub.append(neuron_c)
                    y_sub = torch.cat((y_sub, y_i), 1)

                # set outputs of this level as dataset
                data_tmp = Dataset(f"{data}_level{int(j + 1)}", y_sub, data.Y, data.task)
                rules_tree.append(rules_sub)
            else:
                # set bottom level
                kwargs['data'] = data_tmp
                kwargs['n_rules'] = int(n_rules_tree[j][0])
                neuron_seed.clear()
                neuron = neuron_seed.clone()
                neuron.forward(**kwargs)

                # set bottom level neuron
                rules_btm: List[type(neuron_seed)] = [neuron]
                rules_tree.append(rules_btm)

        self.set_neuron_tree(rules_tree)

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            fea_seperator = FeaSeperator(data.name).get_seperator()
        else:
            fea_seperator = seperator_p.get_seperator()
        data_tmp = data

        for i in torch.arange(len(neuron_tree) - 1):
            output_sub = torch.empty(data.Y.shape[0], 0).double()
            # get level i
            rules_sub: List[type(neuron_seed)] = neuron_tree[int(i)]
            sub_dataset_list = data_tmp.get_subset_fea(fea_seperator[int(i)])
            for j in torch.arange(len(rules_sub)):
                neuron = rules_sub[int(j)]
                sub_dataset_j = sub_dataset_list[j]
                output_j = neuron.predict(sub_dataset_j)
                output_sub = torch.cat((output_sub, output_j), 1)

            # set outputs of this level as dataset
            data_tmp = Dataset(f"{data}_level{int(i + 1)}", output_sub, data.Y, data.task)

        # get bottom neuron
        neuron_btm = neuron_tree[len(neuron_tree) - 1][0]
        y_hat = neuron_btm.predict(data_tmp)
        return y_hat

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree


class TreeFNNet(NetBase):
    """
    the bottome layer is a fc layer instead of a neuron
    """
    def __init__(self, neuron_seed: Neuron):
        super(TreeFNNet, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []
        self.__fc_w: torch.Tensor = []

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator: FeaSeperator = kwargs['seperator']
        fea_seperator = seperator.get_seperator()
        n_rules_tree = seperator.get_n_rule_tree()
        data_tmp = data
        neuron_seed = self.get_neuron_seed()

        rules_tree: List[List[type(neuron_seed)]] = []
        for j in torch.arange(len(fea_seperator)):
            sub_seperator = fea_seperator[int(j)]
            if len(sub_seperator):
                sub_dataset_list = data_tmp.get_subset_fea(sub_seperator)
                # set level j
                rules_sub: List[type(neuron_seed)] = []
                y_sub = torch.empty(data.Y.shape[0], 0).double()
                for i in torch.arange(len(sub_seperator)):
                    neuron_seed.clear()
                    neuron_c = neuron_seed.clone()
                    sub_dataset_i = sub_dataset_list[i]
                    kwargs['data'] = sub_dataset_i
                    kwargs['n_rules'] = int(n_rules_tree[j][i])
                    neuron_c.forward(**kwargs)
                    y_i = neuron_c.predict(sub_dataset_i)
                    rules_sub.append(neuron_c)
                    y_sub = torch.cat((y_sub, y_i), 1)

                # set outputs of this level as dataset
                data_tmp = Dataset(f"{data}_level{int(j + 1)}", y_sub, data.Y, data.task)
                rules_tree.append(rules_sub)
            else:
                # set bottom level fc
                para_mu = kwargs['para_mu']
                w = cal_fc_w(data_tmp.X, data_tmp.Y, para_mu)
                self.set_fc_w(w)

        self.set_neuron_tree(rules_tree)

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)
        self.set_fc_w(torch.empty(0))

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            fea_seperator = FeaSeperator(data.name).get_seperator()
        else:
            fea_seperator = seperator_p.get_seperator()
        data_tmp = data

        for i in torch.arange(len(neuron_tree)):
            output_sub = torch.empty(data.Y.shape[0], 0).double()
            # get level i
            rules_sub: List[type(neuron_seed)] = neuron_tree[int(i)]
            sub_dataset_list = data_tmp.get_subset_fea(fea_seperator[int(i)])
            for j in torch.arange(len(rules_sub)):
                neuron = rules_sub[int(j)]
                sub_dataset_j = sub_dataset_list[j]
                output_j = neuron.predict(sub_dataset_j)
                output_sub = torch.cat((output_sub, output_j), 1)

            # set outputs of this level as dataset
            data_tmp = Dataset(f"{data}_level{int(i + 1)}", output_sub, data.Y, data.task)

        # get bottom fc layer
        w = self.get_fc_w()
        y_hat = data_tmp.X.mm(w)
        return y_hat

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree

    def set_fc_w(self, fc_w: torch.Tensor):
        self.__fc_w = fc_w

    def get_fc_w(self):
        return self.__fc_w


class FnnAO(NetBase):
    """
    the bottome layer is a fc layer instead of a neuron
    """
    def __init__(self, neuron_seed: Neuron):
        super(FnnAO, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []
        self.__w_x: torch.Tensor = []
        self.__w_y: torch.Tensor = []

    def forward(self, **kwargs):
        train_data: Dataset = kwargs['train_data']
        test_data: Dataset = kwargs['test_data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(train_data.name).get_seperator()
        else:
            seperator: FeaSeperator = kwargs['seperator']

        para_mu = kwargs['para_mu']
        para_mu1 = kwargs['para_mu1']

        fea_seperator = seperator.get_seperator()
        seperator.generate_n_rule_tree(kwargs['n_rules'])
        n_rules_tree = seperator.get_n_rule_tree()
        neuron_seed = self.get_neuron_seed()

        sub_seperator = fea_seperator[int(0)]
        sub_dataset_list = train_data.get_subset_fea(sub_seperator)

        sub_dataset_tmp = sub_dataset_list[0]
        n_rule_tmp = int(n_rules_tree[0][0])

        n_smpl = sub_dataset_tmp.X.shape[0]
        n_fea_tmp = sub_dataset_tmp.X.shape[1] + 1
        n_h = n_rule_tmp * n_fea_tmp

        h_all = torch.empty(0, n_smpl, n_h).double()
        n_branch = len(sub_seperator)

        # get neuron tree
        rules_tree: List[List[type(neuron_seed)]] = []
        rules_sub: List[type(neuron_seed)] = []

        # get output of every branches of upper dfnn layer
        for i in torch.arange(n_branch):
            neuron_seed.clear()
            neuron_c = neuron_seed.clone()
            sub_dataset_i = sub_dataset_list[i]
            kwargs['data'] = sub_dataset_i
            kwargs['n_rules'] = int(n_rules_tree[0][i])
            neuron_c.forward(**kwargs)
            rules_sub.append(neuron_c)
            # get rules in neuron and update centers and bias
            rule_ao = neuron_c.get_rules()

            # get h computer in neuron
            h_computer_ao = neuron_c.get_h_computer()
            h_tmp, _ = h_computer_ao.comute_h(sub_dataset_i.X, rule_ao)

            h_cal_tmp = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)

            h_cal_tmp: torch.Tensor = h_cal_tmp.reshape(n_smpl, n_h)
            h_all = torch.cat((h_all, h_cal_tmp.unsqueeze(0)), 0)

        rules_tree.append(rules_sub)
        self.set_neuron_tree(rules_tree)

        # set bottom level AO
        n_midle_output = kwargs['n_hidden_output']
        w_x = torch.rand(n_branch, n_midle_output, n_h).double()
        w_y = torch.rand(n_branch * n_midle_output, 1).double()

        # start AO optimization
        diff = 1
        loss = 100
        train_loss_list = []
        run_th = 0.00001
        n_epoch = 20
        run_epoch = 1

        loss_test_old = 100

        # get h_all
        h_brunch_all = torch.empty(n_smpl, 0).double()
        for i in torch.arange(n_branch):
            h_brunch = h_all[i, :, :].repeat(1, n_midle_output)
            h_brunch_all = torch.cat((h_brunch_all, h_brunch), 1)

        # load better parameters
        data_save_dir = f"./para_file/{kwargs['dataset_name']}"

        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
        para_file = f"{data_save_dir}/ao_{kwargs['n_rules']}_{kwargs['n_hidden_output']}.pt"
        # if os.path.exists(para_fi."]

        g_w_x_best = None
        g_w_y_best = None
        # while diff > run_th and run_epoch < n_epoch:
        while run_epoch < n_epoch:
            # fix  w_y update w_x
            w_y_brunch_all = w_y.repeat(1, n_h)
            w_y_brunch_all = w_y_brunch_all.reshape(1, -1)

            w_x_h_brunch = torch.mul(h_brunch_all, w_y_brunch_all)
            w_x_brunch = torch.inverse(w_x_h_brunch.t().mm(w_x_h_brunch) +
                                       para_mu * torch.eye(w_x_h_brunch.shape[1])
                                       .double()).mm(w_x_h_brunch.t().mm(train_data.Y))
            w_x_cal = w_x_brunch.squeeze().reshape(n_branch, n_midle_output * n_h)

            w_x = w_x_cal.reshape(n_branch, n_midle_output, n_h)

            # fix  w_x update w_y
            w_y_h_cal = torch.empty(n_smpl, 0).double()
            for i in torch.arange(n_branch):
                w_y_h_brunch = h_all[i, :, :].mm(w_x[i, :, :].t()).squeeze()
                if n_midle_output == 1:
                    w_y_h_cal = torch.cat((w_y_h_cal, w_y_h_brunch.unsqueeze(1)), 1)
                else:
                    w_y_h_cal = torch.cat((w_y_h_cal, w_y_h_brunch), 1)

            w_y = torch.inverse(w_y_h_cal.t().mm(w_y_h_cal) + para_mu1 * torch.eye(w_y_h_cal.shape[1])
                                .double()).mm(w_y_h_cal.t().mm(train_data.Y))

            # compute loss
            y_tmp = train_data.Y

            y_hap_tmp = w_y_h_cal.mm(w_y)

            # loss_tmp = torch.norm(y_tmp - y_hap_tmp)
            loss_tmp = mean_squared_error(y_tmp, y_hap_tmp)
            diff = abs(loss_tmp - loss)
            loss = loss_tmp
            train_loss_list.append(loss)

            self.__w_x = w_x
            self.__w_y = w_y
            # snoop the test data performance
            test_y = self.predict(test_data, kwargs['seperator'])
            loss_train = mean_squared_error(y_tmp, y_hap_tmp)
            loss_test = mean_squared_error(test_y, test_data.Y)
            # print(f"Loss of test: {loss_test}")

            if loss_test < loss_test_old and loss_test >= loss_train:
                g_w_x_best = w_x
                g_w_y_best = w_y

            run_epoch = run_epoch + 1
            # print(f"Loss of AO: {loss}  w_y: {w_y.squeeze()[1:5]}")

        # x = torch.linspace(1, len(train_loss_list) + 1, len(train_loss_list)).numpy()
        # y = train_loss_list
        # plt.title('Result Analysis')
        # plt.plot(x, y, color='green', label='loss value')
        # # plt.plot(x, test_acys, color='red', label='training accuracy')
        # plt.legend()  # 显示图例
        #
        # plt.xlabel('iteration epochs')
        # plt.ylabel('loss value')
        # plt.show()

        self.__w_x = w_x
        self.__w_y = w_y
        if g_w_x_best is not None:
            self.__w_x = g_w_x_best
            self.__w_y = g_w_y_best
            para_dict = dict()
            para_dict["w_x"] = w_x
            para_dict["w_y"] = w_y
            torch.save(para_dict, para_file)

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)
        self.__w_x: torch.Tensor = []
        self.__w_y: torch.Tensor = []

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            fea_seperator = FeaSeperator(data.name).get_seperator()
        else:
            fea_seperator = seperator_p.get_seperator()

        neuron_tmp: Neuron = neuron_tree[0][0]
        n_rule_tmp = int(neuron_tmp.get_rules().n_rules)

        sub_dataset_list = data.get_subset_fea(fea_seperator[0])
        sub_dataset_tmp = sub_dataset_list[0]
        n_smpl = sub_dataset_tmp.X.shape[0]
        n_fea_tmp = sub_dataset_tmp.X.shape[1] + 1
        n_h = n_rule_tmp * n_fea_tmp

        h_all = torch.empty(0, n_smpl, n_h).double()
        rules_sub: List[type(neuron_seed)] = neuron_tree[0]

        n_branch = len(rules_sub)
        for j in torch.arange(n_branch):
            neuron = rules_sub[int(j)]
            sub_dataset_j = sub_dataset_list[j]
            neuron.predict(sub_dataset_j)
            # get rules in neuron and update centers and bias
            rule_ao = neuron.get_rules()

            # get h computer in neuron
            h_computer_ao = neuron.get_h_computer()
            h_tmp, _ = h_computer_ao.comute_h(sub_dataset_j.X, rule_ao)

            h_cal_tmp = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)

            h_cal_tmp = h_cal_tmp.reshape(n_smpl, n_h)
            h_all = torch.cat((h_all, h_cal_tmp.unsqueeze(0)), 0)

        # get bottom layer
        w_x = self.__w_x
        w_y = self.__w_y

        # use  w_x
        w_y_h_cal = torch.empty(n_smpl, 0).double()
        for i in torch.arange(n_branch):
            w_y_h_brunch = h_all[i, :, :].mm(w_x[i, :, :].t()).squeeze()
            if len(w_y_h_brunch.shape)==2:
                w_y_h_cal = torch.cat((w_y_h_cal, w_y_h_brunch), 1)
            else:
                w_y_h_cal = torch.cat((w_y_h_cal, w_y_h_brunch.unsqueeze(1)), 1)

        y_hat = w_y_h_cal.mm(w_y)
        return y_hat

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree


# class FnnAO(NetBase):
#     """
#     the bottome layer is a fc layer instead of a neuron
#     """
#
#     def __init__(self, neuron_seed: Neuron):
#         super(FnnAO, self).__init__(neuron_seed)
#         self.__neuron_tree: List[List[type(neuron_seed)]] = []
#         self.__w_x: torch.Tensor = []
#         self.__w_y: torch.Tensor = []
#
#     def forward(self, **kwargs):
#         data: Dataset = kwargs['train_data']
#         if 'seperator' not in kwargs:
#             seperator = FeaSeperator(data.name).get_seperator()
#         else:
#             seperator: FeaSeperator = kwargs['seperator']
#
#         para_mu = kwargs['para_mu']
#         para_mu1 = kwargs['para_mu1']
#
#         fea_seperator = seperator.get_seperator()
#         n_rules_tree = seperator.get_n_rule_tree()
#         neuron_seed = self.get_neuron_seed()
#
#         sub_seperator = fea_seperator[int(0)]
#         sub_dataset_list = data.get_subset_fea(sub_seperator)
#
#         sub_dataset_tmp = sub_dataset_list[0]
#         n_rule_tmp = int(n_rules_tree[0][0])
#
#         n_smpl_tmp = sub_dataset_tmp.X.shape[0]
#         n_fea_tmp = sub_dataset_tmp.X.shape[1] + 1
#         n_h = n_rule_tmp * n_fea_tmp
#
#         h_all = torch.empty(0, n_smpl_tmp, n_h).double()
#         n_branch = len(sub_seperator)
#
#         # get neuron tree
#         rules_tree: List[List[type(neuron_seed)]] = []
#         rules_sub: List[type(neuron_seed)] = []
#
#         # get output of every branches of upper dfnn layer
#         for i in torch.arange(n_branch):
#             neuron_seed.clear()
#             neuron_c = neuron_seed.clone()
#             sub_dataset_i = sub_dataset_list[i]
#             kwargs['data'] = sub_dataset_i
#             kwargs['n_rules'] = int(n_rules_tree[0][i])
#             neuron_c.forward(**kwargs)
#             rules_sub.append(neuron_c)
#             # get rules in neuron and update centers and bias
#             rule_ao = neuron_c.get_rules()
#
#             # get h computer in neuron
#             h_computer_ao = neuron_c.get_h_computer()
#             h_tmp, _ = h_computer_ao.comute_h(sub_dataset_i.X, rule_ao)
#
#             h_cal_tmp = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)
#
#             h_cal_tmp: torch.Tensor = h_cal_tmp.reshape(n_smpl_tmp, n_h)
#             h_all = torch.cat((h_all, h_cal_tmp.unsqueeze(0)), 0)
#
#         rules_tree.append(rules_sub)
#         self.set_neuron_tree(rules_tree)
#
#         # set bottom level AO
#         w_x = torch.rand(n_branch, n_h).double()
#         w_y = torch.rand(n_branch, 1).double()
#         w_y_h = torch.rand(n_smpl_tmp, n_branch).double()
#         w_x_h = h_all.clone()
#
#         # start AO optimization
#         diff = 1
#         loss = 100
#         run_th = 0.0001
#         run_epoch = 1
#         n_epoch = 300
#         train_loss_list = []
#         # while diff > run_th and run_epoch < n_epoch:
#         while run_epoch < n_epoch:
#             # fix  w_y update w_x
#             for i in torch.arange(n_branch):
#                 w_x_h[i, :, :] = w_y[i] * h_all[i, :, :]
#
#             w_x_h_cal = w_x_h.permute(1, 0, 2)
#             w_x_h_cal = w_x_h_cal.reshape(n_smpl_tmp, -1)
#             w_x_optimal = torch.inverse(w_x_h_cal.t().mm(w_x_h_cal) + para_mu * torch.eye(w_x_h_cal.shape[1]).double()) \
#                 .mm(w_x_h_cal.t().mm(data.Y))
#
#             w_x = w_x_optimal.reshape(n_branch, n_h)
#
#             # fix  w_x update w_y
#             for i in torch.arange(n_branch):
#                 w_y_h[:, i] = h_all[i, :, :].mm(w_x[i, :].unsqueeze(1)).squeeze()
#
#             w_y = torch.inverse(w_y_h.t().mm(w_y_h) + para_mu1 * torch.eye(w_y_h.shape[1]).double()) \
#                 .mm(w_y_h.t().mm(data.Y))
#
#             # compute loss
#             y_tmp = data.Y
#
#             y_hap_tmp = w_y_h.mm(w_y)
#
#             loss_tmp = mean_squared_error(y_tmp, y_hap_tmp)
#             # loss_tmp = torch.norm(y_tmp - y_hap_tmp)
#             diff = abs(loss_tmp - loss)
#             loss = loss_tmp
#             train_loss_list.append(loss)
#             print(f"Loss of AO: {loss}")
#             run_epoch = run_epoch + 1
#
#         x = torch.linspace(1, len(train_loss_list) + 1, len(train_loss_list)).numpy()
#         y = train_loss_list
#         plt.title('Result Analysis')
#         plt.plot(x, y, color='green', label='loss value')
#         # plt.plot(x, test_acys, color='red', label='training accuracy')
#         plt.legend()  # 显示图例
#
#         plt.xlabel('iteration times')
#         plt.ylabel('rate')
#         plt.show()
#         self.__w_x = w_x
#
#         self.__w_y = w_y
#
#     def clear(self):
#         neuron_seed = self.get_neuron_seed()
#         self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
#                                                neuron_seed.get_h_computer(),
#                                                neuron_seed.get_fnn_solver()))
#         neuron_tree: List[List[type(neuron_seed)]] = []
#         self.set_neuron_tree(neuron_tree)
#         self.__w_x: torch.Tensor = []
#         self.__w_y: torch.Tensor = []
#
#     def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
#         neuron_tree = self.get_neuron_tree()
#         neuron_seed = self.get_neuron_seed()
#         if seperator_p is None:
#             fea_seperator = FeaSeperator(data.name).get_seperator()
#         else:
#             fea_seperator = seperator_p.get_seperator()
#
#         neuron_tmp: Neuron = neuron_tree[0][0]
#         n_rule_tmp = int(neuron_tmp.get_rules().n_rules)
#
#         sub_dataset_list = data.get_subset_fea(fea_seperator[0])
#         sub_dataset_tmp = sub_dataset_list[0]
#         n_smpl_tmp = sub_dataset_tmp.X.shape[0]
#         n_fea_tmp = sub_dataset_tmp.X.shape[1] + 1
#         n_h = n_rule_tmp * n_fea_tmp
#
#         h_all = torch.empty(0, n_smpl_tmp, n_h).double()
#         rules_sub: List[type(neuron_seed)] = neuron_tree[0]
#
#         n_branch = len(rules_sub)
#         for j in torch.arange(n_branch):
#             neuron = rules_sub[int(j)]
#             sub_dataset_j = sub_dataset_list[j]
#             neuron.predict(sub_dataset_j)
#             # get rules in neuron and update centers and bias
#             rule_ao = neuron.get_rules()
#
#             # get h computer in neuron
#             h_computer_ao = neuron.get_h_computer()
#             h_tmp, _ = h_computer_ao.comute_h(sub_dataset_j.X, rule_ao)
#
#             h_cal_tmp = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)
#
#             h_cal_tmp = h_cal_tmp.reshape(n_smpl_tmp, n_h)
#             h_all = torch.cat((h_all, h_cal_tmp.unsqueeze(0)), 0)
#
#         # get bottom layer
#         w_x = self.__w_x
#         w_y = self.__w_y
#
#         w_y_h = torch.rand(n_smpl_tmp, n_branch).double()
#         for i in torch.arange(n_branch):
#             w_y_h[:, i] = h_all[i, :, :].mm(w_x[i, :].unsqueeze(1)).squeeze()
#
#         y_hat = w_y_h.mm(w_y)
#         return y_hat
#
#     def get_neuron_tree(self):
#         return self.__neuron_tree
#
#     def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
#         self.__neuron_tree = neuron_tree


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(18, 18),
            nn.Linear(18, 9),
            nn.Linear(9, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1).float()
        x = self.layers(x)
        return x


# class ConsequentNet(nn.Module):
#     def __init__(self, n_brunch, n_h_fea):
#         super(ConsequentNet, self).__init__()
#         self.n_brunch = n_brunch
#         self.n_h_fea = n_h_fea  # number of H matrix features
#
#         n_out_layer1 = n_brunch
#         layer_head = nn.Sequential(
#             nn.Linear(n_brunch*n_h_fea, n_out_layer1),
#             nn.Softmax(),
#         )
#         layer_foot = nn.Sequential(
#             nn.Linear(n_brunch, 2),
#             nn.ReLU(),
#         )
#         self.__layer_head = layer_head
#         self.__layer_foot = layer_foot
#
#     def forward(self, x: torch.Tensor):
#         x = x.view(x.shape[0], -1).float()
#         input_layer_foot = self.__layer_head(x)
#         output_layer_foot = self.__layer_foot(input_layer_foot)
#         return output_layer_foot


# class ConsequentNet(nn.Module):
#     def __init__(self, n_brunch, n_h_fea):
#         super(ConsequentNet, self).__init__()
#         self.n_brunch = n_brunch
#         self.n_h_fea = n_h_fea  # number of H matrix features
#
#         n_out_layer1 = 1
#         layer_head = []
#         for i in torch.arange(n_brunch):
#             layer_head.append(nn.Linear(n_h_fea, n_out_layer1))
#         layer_foot = nn.Sequential(
#             nn.Linear(n_brunch, 2),
#             nn.ReLU(),
#         )
#         self.__layer_head = layer_head
#         self.__layer_foot = layer_foot
#
#     def forward(self, x: torch.Tensor):
#         layer_head = self.__layer_head
#
#         x = x.permute(1, 0, 2)
#         input_layer_foot = torch.empty(x.shape[1], 0).float()
#
#         for i in torch.arange(x.shape[0]):
#             x_sub = x[i, :, :].float()
#             sub_net = layer_head[int(i)]
#             out_layer_head = sub_net(x_sub)
#             input_layer_foot = torch.cat((input_layer_foot, out_layer_head), 1)
#
#         output_layer_foot = self.__layer_foot(input_layer_foot)
#         return output_layer_foot


class ConsequentNet(nn.Module):
    def __init__(self, n_brunch, n_h_fea):
        super(ConsequentNet, self).__init__()
        self.n_brunch = n_brunch
        self.n_h_fea = n_h_fea  # number of H matrix features

        n_out_layer1 = 1
        self.__layer_head_b1 = nn.Linear(n_h_fea, n_out_layer1)
        self.__layer_head_b2 = nn.Linear(n_h_fea, n_out_layer1)
        self.__layer_head_b3 = nn.Linear(n_h_fea, n_out_layer1)
        self.__layer_head_b4 = nn.Linear(n_h_fea, n_out_layer1)
        self.__layer_head_b5 = nn.Linear(n_h_fea, n_out_layer1)
        self.__layer_head_b6 = nn.Linear(n_h_fea, n_out_layer1)
        layer_foot = nn.Sequential(
            nn.Linear(n_brunch, 2),
            nn.ReLU(),
        )

        self.__layer_foot = layer_foot

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)
        input_layer_foot = torch.zeros(x.shape[1], self.n_brunch).float()

        x_sub1 = x[0, :, :].float()
        input_layer_foot[:, 0] = self.__layer_head_b1(x_sub1).squeeze()

        x_sub2 = x[1, :, :].float()
        input_layer_foot[:, 1] = self.__layer_head_b2(x_sub2).squeeze()

        x_sub3 = x[2, :, :].float()
        input_layer_foot[:, 2] = self.__layer_head_b3(x_sub3).squeeze()

        x_sub4 = x[3, :, :].float()
        input_layer_foot[:, 3] = self.__layer_head_b4(x_sub4).squeeze()

        x_sub5 = x[4, :, :].float()
        input_layer_foot[:, 4] = self.__layer_head_b5(x_sub5).squeeze()

        x_sub6 = x[5, :, :].float()
        input_layer_foot[:, 5] = self.__layer_head_b6(x_sub6).squeeze()

        output_layer_foot = self.__layer_foot(input_layer_foot)
        return output_layer_foot


class FnnDnn(NetBase):
    """
    the bottome layer is a fc layer instead of a neuron
    """

    def __init__(self, neuron_seed: Neuron):
        super(FnnDnn, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []
        self.__consequent_net: ConsequentNet = []

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator: FeaSeperator = kwargs['seperator']

        fea_seperator = seperator.get_seperator()
        n_rules_tree = seperator.get_n_rule_tree()
        neuron_seed = self.get_neuron_seed()

        sub_seperator = fea_seperator[int(0)]
        sub_dataset_list = data.get_subset_fea(sub_seperator)

        sub_dataset_tmp = sub_dataset_list[0]
        n_rule_tmp = int(n_rules_tree[0][0])

        n_smpl = sub_dataset_tmp.X.shape[0]
        n_fea_tmp = sub_dataset_tmp.X.shape[1] + 1
        n_h = n_rule_tmp * n_fea_tmp

        h_all = torch.empty(0, n_smpl, n_h).double()
        n_branch = len(sub_seperator)

        # get neuron tree
        rules_tree: List[List[type(neuron_seed)]] = []
        rules_sub: List[type(neuron_seed)] = []

        # get output of every branches of upper dfnn layer
        for i in torch.arange(n_branch):
            neuron_seed.clear()
            neuron_c = neuron_seed.clone()
            sub_dataset_i = sub_dataset_list[i]
            kwargs['data'] = sub_dataset_i
            kwargs['n_rules'] = int(n_rules_tree[0][i])
            neuron_c.forward(**kwargs)
            rules_sub.append(neuron_c)
            # get rules in neuron and update centers and bias
            rule_ao = neuron_c.get_rules()

            # get h computer in neuron
            h_computer_ao = neuron_c.get_h_computer()
            h_tmp, _ = h_computer_ao.comute_h(sub_dataset_i.X, rule_ao)

            h_cal_tmp = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)

            h_cal_tmp: torch.Tensor = h_cal_tmp.reshape(n_smpl, n_h)
            h_all = torch.cat((h_all, h_cal_tmp.unsqueeze(0)), 0)

        rules_tree.append(rules_sub)
        self.set_neuron_tree(rules_tree)

        # set consequent net
        consequent_net: ConsequentNet = ConsequentNet(n_branch, n_h)
        optimizer = torch.optim.Adam(consequent_net.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        # transform h tensor to torch dataset
        train_dataset = DatasetH(x=h_all, y=data.Y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=n_smpl, shuffle=False)

        epochs = 3000
        train_th = 0.000001
        train_losses = []
        for epoch in range(epochs):
            consequent_net.train()
            for i, (h_batch, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = consequent_net(h_batch)
                loss = loss_fn(outputs.double(), labels.long().squeeze(1))
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                print(f"epoch : {epoch + 1}, train loss : {train_losses[-1]} ")
            # if len(train_losses) >= 2:
            #     loss_diff = abs(train_losses[-1] - train_losses[len(train_losses)-2])
            #     # when the whole model stops to update
            #     if loss_diff < train_th:
            #         break

            # validate the model
            consequent_net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (h_batch, labels) in enumerate(train_loader):
                    outputs = consequent_net(h_batch)
                    # loss = loss_fn(outputs, labels.long().squeeze(1))

                    # valid_losses = loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels.squeeze().long()).sum().item()
                    total += labels.size(0)

                    # print(f"valid loss : {valid_losses}")

            accuracy = 100 * correct / total
            print(f"train acc : {accuracy}%")
        self.__consequent_net = consequent_net

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)
        self.__consequent_net = []

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            fea_seperator = FeaSeperator(data.name).get_seperator()
        else:
            fea_seperator = seperator_p.get_seperator()

        neuron_tmp: Neuron = neuron_tree[0][0]
        n_rule_tmp = int(neuron_tmp.get_rules().n_rules)

        sub_dataset_list = data.get_subset_fea(fea_seperator[0])
        sub_dataset_tmp = sub_dataset_list[0]
        n_smpl = sub_dataset_tmp.X.shape[0]
        n_fea_tmp = sub_dataset_tmp.X.shape[1] + 1
        n_h = n_rule_tmp * n_fea_tmp

        h_all = torch.empty(0, n_smpl, n_h).double()
        rules_sub: List[type(neuron_seed)] = neuron_tree[0]

        n_branch = len(rules_sub)
        for j in torch.arange(n_branch):
            neuron = rules_sub[int(j)]
            sub_dataset_j = sub_dataset_list[j]
            neuron.predict(sub_dataset_j)
            # get rules in neuron and update centers and bias
            rule_ao = neuron.get_rules()

            # get h computer in neuron
            h_computer_ao = neuron.get_h_computer()
            h_tmp, _ = h_computer_ao.comute_h(sub_dataset_j.X, rule_ao)

            h_cal_tmp = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)

            h_cal_tmp = h_cal_tmp.reshape(n_smpl, n_h)
            h_all = torch.cat((h_all, h_cal_tmp.unsqueeze(0)), 0)

        # get consequent net
        # transform h tensor to torch dataset
        test_dataset = DatasetH(x=h_all, y=data.Y)
        test_loader = DataLoader(dataset=test_dataset, batch_size=n_smpl, shuffle=False)
        consequent_net = self.__consequent_net

        consequent_net.eval()
        y_hat = torch.empty(0)
        with torch.no_grad():
            for i, (h_batch, labels) in enumerate(test_loader):
                outputs = consequent_net(h_batch)
                _, y_hat_tmp = torch.max(outputs.data, 1)
                y_hat = torch.cat((y_hat, y_hat_tmp.unsqueeze(0).float()), 1)

        return y_hat.t().double()

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree


class TreeDeepNet(NetBase):
    """
    the bottome layer is a deep net instead of a neuron
    """
    def __init__(self, neuron_seed: Neuron):
        super(TreeDeepNet, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []
        self.__lei_net: ConsequentNet = []

    def set_lei_net(self, lei_net: ConsequentNet):
        self.__lei_net = lei_net

    def get_lei_net(self):
        return self.__lei_net

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator: FeaSeperator = kwargs['seperator']

        fea_seperator = seperator.get_seperator()
        n_rules_tree = seperator.get_n_rule_tree()
        data_tmp = data
        neuron_seed = self.get_neuron_seed()

        output = None
        w_sub = []
        rules_tree: List[List[type(neuron_seed)]] = []
        for j in torch.arange(len(fea_seperator)):
            sub_seperator = fea_seperator[int(j)]

            if len(sub_seperator):
                sub_dataset_list = data_tmp.get_subset_fea(sub_seperator)
                # get the input of below net work
                rules_sub: List[type(neuron_seed)] = []
                for i in torch.arange(len(sub_seperator)):
                    neuron_seed.clear()
                    neuron_c = neuron_seed.clone()
                    sub_dataset_i = sub_dataset_list[i]
                    kwargs['data'] = sub_dataset_i
                    kwargs['n_rules'] = int(n_rules_tree[j][i])
                    neuron_c.forward(**kwargs)
                    w_sub.append(neuron_c.get_rules().consequent_list)
                    rules_sub.append(neuron_c)
                rules_tree.append(rules_sub)

            else:
                # creat neural net work
                loss_fn = nn.CrossEntropyLoss()

                n_neuron = len(fea_seperator[0])
                n_rules = int(n_rules_tree[0][0])
                model_foot: ConsequentNet = ConsequentNet(n_neuron, n_rules)
                optimizer = torch.optim.Adam(model_foot.parameters(), lr=0.0001)

                epochs = 1500
                train_losses = []
                for epoch in range(epochs):
                    model_foot.train()
                    outputs = model_foot(w_sub)
                    loss = loss_fn(outputs.double(), data.Y.long().squeeze(1))
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

                    print(f"epoch : {epoch + 1}, train loss : {train_losses[-1]} ")

                    # validate the model
                    model_foot.eval()
                    with torch.no_grad():
                        outputs = model_foot(w_sub)
                        loss = loss_fn(outputs, data.Y.long().squeeze(1))

                        valid_losses = loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        correct = (predicted == data.Y.squeeze().long()).sum().item()
                        total = data.Y.size(0)

                    accuracy = 100 * correct / total
                    print(f"valid loss : {valid_losses}, valid acc : {accuracy}%")
                self.set_lei_net(model_foot)
        self.set_neuron_tree(rules_tree)
        return output

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)
        self.__lei_net = []

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            fea_seperator = FeaSeperator(data.name).get_seperator()
        else:
            fea_seperator = seperator_p.get_seperator()
        data_tmp = data

        w_sub = []
        for i in torch.arange(len(neuron_tree)):
            # get level i

            rules_sub: List[type(neuron_seed)] = neuron_tree[int(i)]
            sub_dataset_list = data_tmp.get_subset_fea(fea_seperator[int(i)])
            for j in torch.arange(len(rules_sub)):
                neuron = rules_sub[int(j)]
                sub_dataset_j = sub_dataset_list[j]
                output_j = neuron.predict(sub_dataset_j)
                w_sub.append(output_j)

        # get bottom fc layer
        loss_fn = nn.CrossEntropyLoss()

        model: ConsequentNet = self.get_lei_net()
        model.eval()
        with torch.no_grad():
            outputs = model(w_sub)
            loss = loss_fn(outputs, data.Y.long().squeeze(1))

            valid_losses = loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == data.Y.squeeze().long()).sum().item()
            total = data.Y.size(0)

        accuracy = 100 * correct / total
        print(f"test loss : {valid_losses}, test acc : {accuracy}%")

        return accuracy

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree


class FnnNet(NetBase, nn.Module):
    """
    replace consequent layer with DNN net structure
    """
    def __init__(self, neuron_seed: Neuron):
        super(FnnNet, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []
        self.__fc_w: torch.Tensor = []

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator: FeaSeperator = kwargs['seperator']
        fea_seperator = seperator.get_seperator()
        n_rules_tree = seperator.get_n_rule_tree()
        data_tmp = data
        neuron_seed = self.get_neuron_seed()

        rules_tree: List[List[type(neuron_seed)]] = []
        for j in torch.arange(len(fea_seperator)):
            sub_seperator = fea_seperator[int(j)]
            if len(sub_seperator):
                sub_dataset_list = data_tmp.get_subset_fea(sub_seperator)
                # set level j
                rules_sub: List[type(neuron_seed)] = []
                y_sub = torch.empty(data.Y.shape[0], 0).double()
                for i in torch.arange(len(sub_seperator)):
                    neuron_seed.clear()
                    neuron_c = neuron_seed.clone()
                    sub_dataset_i = sub_dataset_list[i]
                    kwargs['data'] = sub_dataset_i
                    kwargs['n_rules'] = int(n_rules_tree[j][i])
                    neuron_c.forward(**kwargs)
                    y_i = neuron_c.predict(sub_dataset_i)
                    rules_sub.append(neuron_c)
                    y_sub = torch.cat((y_sub, y_i), 1)

                # set outputs of this level as dataset
                data_tmp = Dataset(f"{data}_level{int(j + 1)}", y_sub, data.Y, data.task)
                rules_tree.append(rules_sub)
            else:
                # set bottom level fc
                para_mu = kwargs['para_mu']
                w = cal_fc_w(data_tmp.X, data_tmp.Y, para_mu)
                self.set_fc_w(w)

        self.set_neuron_tree(rules_tree)

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)
        self.set_fc_w(torch.empty(0))

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            fea_seperator = FeaSeperator(data.name).get_seperator()
        else:
            fea_seperator = seperator_p.get_seperator()
        data_tmp = data

        for i in torch.arange(len(neuron_tree)):
            output_sub = torch.empty(data.Y.shape[0], 0).double()
            # get level i
            rules_sub: List[type(neuron_seed)] = neuron_tree[int(i)]
            sub_dataset_list = data_tmp.get_subset_fea(fea_seperator[int(i)])
            for j in torch.arange(len(rules_sub)):
                neuron = rules_sub[int(j)]
                sub_dataset_j = sub_dataset_list[j]
                output_j = neuron.predict(sub_dataset_j)
                output_sub = torch.cat((output_sub, output_j), 1)

            # set outputs of this level as dataset
            data_tmp = Dataset(f"{data}_level{int(i + 1)}", output_sub, data.Y, data.task)

        # get bottom fc layer
        w = self.get_fc_w()
        y_hat = data_tmp.X.mm(w)
        return y_hat

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree

    def set_fc_w(self, fc_w: torch.Tensor):
        self.__fc_w = fc_w

    def get_fc_w(self):
        return self.__fc_w
