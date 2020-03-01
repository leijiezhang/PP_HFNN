from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_rules_para_kfold
from utils import load_data, Logger
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('hrss_config')

para_mu_list = torch.arange(-10, 1, 1).double()
param_config.para_mu_list = torch.pow(2, para_mu_list).double()
param_config.para_mu1_list = torch.pow(2, para_mu_list).double()

n_rule_list = torch.arange(1, 11, 1)
param_config.n_rules_list = n_rule_list

acc_c_train_arr = []
acc_c_test_arr = []
acc_d_train_arr = []
acc_d_test_arr = []

acc_c_train_list = []
acc_c_test_list = []
acc_d_train_list = []
acc_d_test_list = []
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file, param_config.dataset_name)
    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    acc_c_train_tsr, acc_c_test_tsr, acc_d_train_tsr, acc_d_test_tsr = \
        dfnn_rules_para_kfold(param_config, dataset)

    acc_c_train_list.append(acc_c_train_tsr)
    acc_c_test_list.append(acc_c_test_tsr)
    acc_d_train_list.append(acc_d_train_tsr)
    acc_d_test_list.append(acc_d_test_tsr)

    loss_c_train_mean_mtrx = acc_c_train_tsr.mean(2)
    loss_c_test_mean_mtrx = acc_c_test_tsr.mean(2)
    loss_d_train_mean_mtrx = acc_d_train_tsr.mean(2)
    loss_d_test_mean_mtrx = acc_d_test_tsr.mean(2)

    loss_c_train_std_mtrx = acc_c_train_tsr.std(2)
    loss_c_test_std_mtrx = acc_c_test_tsr.std(2)
    loss_d_train_std_mtrx = acc_d_train_tsr.std(2)
    loss_d_test_std_mtrx = acc_d_test_tsr.std(2)

    acc_c_test = loss_c_test_mean_mtrx.max()
    best_c_mask = torch.eq(loss_c_test_mean_mtrx, acc_c_test)

    acc_c_train = loss_c_train_mean_mtrx[best_c_mask].max()
    acc_c_train_std = loss_c_train_std_mtrx[best_c_mask]
    acc_c_test_std = loss_c_test_std_mtrx[best_c_mask]

    acc_d_test = loss_d_test_mean_mtrx.max()
    best_d_mask = torch.eq(loss_d_test_mean_mtrx, acc_d_test)
    acc_d_train = loss_d_train_mean_mtrx[best_d_mask].max()
    acc_d_train_std = loss_d_train_std_mtrx[best_d_mask]
    acc_d_test_std = loss_d_test_std_mtrx[best_d_mask]

    acc_c_train_arr.append([acc_c_train, acc_c_train_std])
    acc_c_test_arr.append([acc_c_test, acc_c_test_std])
    acc_d_train_arr.append([acc_d_train, acc_d_train_std])
    acc_d_test_arr.append([acc_d_test, acc_d_test_std])

    param_config.log.info(
        f"mAp of training data on centralized method: "
        f"{round(float(acc_c_train), 4)}/{round(float(acc_c_train_std), 4)}")
    param_config.log.info(
        f"mAp of test data on centralized method: "
        f"{round(float(acc_c_test), 4)}/{round(float(acc_c_test_std), 4)}")
    param_config.log.info(
        f"mAp of training data on distributed method: "
        f"{round(float(acc_d_train), 4)}/{round(float(acc_d_train_std), 4)}")
    param_config.log.info(
        f"mAp of test data on distributed method:"
        f" {round(float(acc_d_test), 4)}/{round(float(acc_d_test_std), 4)}")

save_dict = dict()
save_dict["acc_c_train_list"] = acc_c_train_list
save_dict["acc_c_test_list"] = acc_c_test_list
save_dict["acc_d_train_list"] = acc_d_train_list
save_dict["acc_d_test_list"] = acc_d_test_list

save_dict["acc_c_train_arr"] = acc_c_train_arr
save_dict["acc_c_test_arr"] = acc_c_test_arr
save_dict["acc_d_train_arr"] = acc_d_train_arr
save_dict["acc_d_test_arr"] = acc_d_test_arr

data_save_dir = f"./results/{param_config.dataset_name}"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/{param_config.model_name}_{param_config.n_rules}.pt"
torch.save(save_dict, data_save_file)
