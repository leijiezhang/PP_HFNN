import torch
from utils import dataset_parse, eeg_dataset_parse, seed_dataset_parse


# # parse seed dataset
# trial_num_list = [15, 30]
channel_list_all = []
channel_list = ['FT7', 'FT8', 'T7', 'T8']
channel_list_all.append(channel_list)
channel_list = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8']
channel_list_all.append(channel_list)
channel_list = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'FP1', 'FPZ', 'FP2']
channel_list_all.append(channel_list)
channel_list = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'C5', 'C6', 'CP5', 'CP6']
channel_list_all.append(channel_list)
channel_list = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 
                'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 
                'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 
                'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 
                'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
channel_list_all.append(channel_list)

# # data_list = ['HRSS_anomalous_optimized']
#
# for k in torch.arange(len(trial_num_list)),
#     for i in torch.arange(len(channel_list_all)),
#         for j in torch.arange(15),
#             dataset_name = f"seed_cnum{len(channel_list_all[int(i)])}_subj{j + 1}"
#             sub_fold = f"seed_trial{trial_num_list[int(k)]}"
#             eeg_dataset_parse(dataset_name, sub_fold)
# #
#
# for k in torch.arange(len(trial_num_list)),
#     for i in torch.arange(len(channel_list_all)),
#         dataset_name = f"seed_cnum{len(channel_list_all[int(i)])}_all"
#         sub_fold = f"seed_trial{trial_num_list[int(k)]}"
#         eeg_dataset_parse(dataset_name, sub_fold)


# parse eeg dual dataset
dataset_name = f"ethylene_meth"
sub_fold = "ethylene"
# eeg_dataset_parse(dataset_name, sub_fold)
dataset_parse(dataset_name, sub_fold)
# for i in torch.arange(11):
#     dataset_name = f"eegDual_subj{i+1}"
#     sub_fold = "eeg_dual"
#     # eeg_dataset_parse(dataset_name, sub_fold)
#     dataset_parse(dataset_name, sub_fold)

# # parse xiaofei dataset
# for i in torch.arange(8),
#     dataset_name = f"xiaofei_subj{i+1}"
#     sub_fold = "xiaofei"
#     eeg_dataset_parse(dataset_name, sub_fold)

# # parse seed dataset
# for i in torch.arange(len(channel_list_all)):
#     for k in torch.arange(3):
#         for j in torch.arange(15):
#             seed_dataset_parse(len(channel_list_all[int(i)]), int(k+1), int(j+1))

# # parse eeg sleep dataset
# for i in torch.arange(5):
#     dataset_name = f"eegSleep_subj{i+1}"
#     sub_fold = "eeg_sleep"
#     dataset_parse(dataset_name, sub_fold)
