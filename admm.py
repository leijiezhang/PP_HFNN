class ADMM(object):
    """

    """
    def __init__(self, consensus_max_steps=300, consensus_thres=0.001, admm_max_steps=300,
                 admm_rho=1, admm_reltol=0.001,admm_abstol=0.001):
        self.consensus_max_steps = consensus_max_steps
        self.consensus_thres = consensus_thres
        self.admm_max_steps = admm_max_steps
        self.admm_rho = admm_rho
        self.admm_reltol = admm_reltol
        self.admm_abstol = admm_abstol


    # def solve(self, alg, net, trainData: