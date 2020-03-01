from h_utils import HBase, HNormal
from fnn_solver import FnnSolveBase, FnnSolveReg, FnnSolveCls
from rules import RuleBase, RuleKmeans, RuleFuzzyCmeans
from partition import PartitionStrategy, KFoldPartition
from loss_utils import LossFunc, NRMSELoss, RMSELoss, MSELoss, Map, LikelyLoss
from utils import Logger
from neuron import Neuron
from seperator import FeaSeperator
from model import NetBase, TreeNet, TreeFNNet, FnnDnn, FnnAO
import yaml


class ParamConfig(object):
    def __init__(self, n_run=1, n_kfolds=10, n_agents=25, nrules=10):
        self.n_run = n_run  # Number of simulations
        self.n_kfolds = n_kfolds  # Number of folds

        # Network configuration
        self.n_agents = n_agents
        self.n_agents_list = []

        self.connectivity = 0.25  # Connectivity in the networks(must be between 0 and 1)

        self.n_rules = nrules  # number of rules in stage 1
        self.n_rules_list = []

        self.n_hidden_output = 1

        self.dataset_list_all = []
        self.dataset_list = ['CASP']
        self.dataset_name = 'hrss'

        # set mu
        self.para_mu_current = 0
        self.para_mu_list = []
        self.para_mu1_current = 0
        self.para_mu1_list = []
        # set rho
        self.para_rho = 1

        # initiate tools
        self.h_computer: HBase = None
        self.fnn_solver: FnnSolveBase = None
        self.loss_fun: LossFunc = None
        self.rules: RuleBase = None
        self.patition_strategy: PartitionStrategy = None

        # set feature seperator
        self.fea_seperator: FeaSeperator = None

        # initiate net
        self.model: NetBase = None
        self.model_name = ''
        self.log = None

        # config content
        self.config_content = None

    def config_parse(self, config_name):
        config_dir = f"./configs/{config_name}.yaml"
        config_file = open(config_dir)
        config_content = yaml.load(config_file, Loader=yaml.FullLoader)
        self.config_content = config_content

        self.n_run = config_content['n_run']
        self.n_kfolds = config_content['n_kfolds']

        self.n_agents = config_content['n_agents']
        self.n_agents_list = config_content['n_agents_list']

        self.n_rules = config_content['n_rules']
        self.n_rules_list = config_content['n_rules_list']

        self.n_hidden_output = config_content['n_hidden_output']

        self.dataset_list_all = config_content['dataset_list_all']
        self.dataset_list = config_content['dataset_list']
        self.dataset_name = config_content['dataset_name']

        self.para_mu_current = config_content['mu_current']
        self.para_mu_list = config_content['mu_list']
        self.para_mu1_current = config_content['mu1_current']
        self.para_mu1_list = config_content['mu1_list']

        self.para_rho = config_content['rho']

        # set h_computer
        if config_content['h_computer'] == 'base':
            self.h_computer = HBase()
        elif config_content['h_computer'] == 'normal':
            self.h_computer = HNormal()

        # set fnn_solver
        if config_content['fnn_solver'] == 'base':
            self.fnn_solver = FnnSolveBase()
        elif config_content['fnn_solver'] == 'normal':
            self.fnn_solver = FnnSolveReg()
        elif config_content['fnn_solver'] == 'sigmoid':
            self.fnn_solver = FnnSolveCls()

        # set loss_fun:loss function
        if config_content['loss_fun'] == 'base':
            self.loss_fun = LossFunc()
        elif config_content['loss_fun'] == 'rmse':
            self.loss_fun = RMSELoss()
        elif config_content['loss_fun'] == 'nrmse':
            self.loss_fun = NRMSELoss()
        elif config_content['loss_fun'] == 'mse':
            self.loss_fun = MSELoss()
        elif config_content['loss_fun'] == 'map':
            self.loss_fun = Map()
        elif config_content['loss_fun'] == 'likely':
            self.loss_fun = LikelyLoss()

        # set rules
        if config_content['rules'] == 'base':
            self.rules = RuleBase()
        elif config_content['rules'] == 'kmeans':
            self.rules = RuleKmeans()
        elif config_content['rules'] == 'fuzzyc':
            self.rules = RuleFuzzyCmeans()

        # set patition_strategy
        if config_content['rules'] == 'kmeans':
            self.patition_strategy = KFoldPartition(self.n_kfolds)

        # set logger to decide whether write log into files
        if config_content['log_to_file'] == 'false':
            self.log = Logger()
        else:
            self.log = Logger(True)

        # set model
        neuron = Neuron(self.rules, self.h_computer, self.fnn_solver)
        self.model_name = config_content['model']
        if config_content['model'] == 'base':
            self.model = NetBase(neuron)
        elif config_content['model'] == 'hdfnn_fn':
            self.model = TreeFNNet(neuron)
        elif config_content['model'] == 'hdfnn':
            self.model = TreeNet(neuron)
            tree_rule_spesify = config_content['tree_rule_spesify']
            if tree_rule_spesify == 'true':
                n_rule_spesify = config_content['n_rule_spesify']
                self.model_name = f"{self.model_name}_s_{n_rule_spesify}"
        elif config_content['model'] == 'hdfnn_dnn':
            self.model = FnnDnn(neuron)
        elif config_content['model'] == 'hdfnn_ao':
            self.model = FnnAO(neuron)

    def get_cur_dataset(self, dataset_idx):
        dataset_name = self.dataset_list[dataset_idx]
        config_content = self.config_content

        fea_seperator = FeaSeperator(dataset_name)
        # set feature seperator
        seperator_type = config_content['feature_seperator']
        if seperator_type == 'slice_window':
            window_size = config_content['window_size']
            step = config_content['step']
            n_level = config_content['n_level']
            fea_seperator.set_seperator_by_slice_window(window_size, step, n_level)
        elif seperator_type == 'stride_window':
            stride_len = config_content['stride_len']
            n_level = config_content['n_level']
            fea_seperator.set_seperator_by_stride_window(stride_len, n_level)
        elif seperator_type == 'random_pick':
            window_size = config_content['window_size']
            n_repeat = config_content['n_repeat_select']
            n_level = config_content['n_level']
            fea_seperator.set_seperator_by_random_pick(window_size, n_repeat,
                                                       n_level)
        elif seperator_type == 'no_seperate':
            fea_seperator.set_seperator_by_no_seperate()

        # generate tree of rule numbers according to the seperator
        fea_seperator.generate_n_rule_tree(self.n_rules)
        # set rule number
        tree_rule_spesify = config_content['tree_rule_spesify']
        if tree_rule_spesify == 'true':
            n_rule_pos = config_content['n_rule_pos']
            n_rule_spesify = config_content['n_rule_spesify']
            fea_seperator.set_n_rule_tree(n_rule_pos[0], n_rule_pos[1], n_rule_spesify)
        self.fea_seperator = fea_seperator
        return dataset_name
