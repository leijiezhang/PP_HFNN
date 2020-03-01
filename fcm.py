import torch
from sklearn.metrics.pairwise import euclidean_distances
"""
this file is useless, just for verifying the sk-fuzzy package
I didn't use this file in the whole training process
"""


SMALL_VALUE = 0.00001


class FCM:
    """
        This algorithm is from the paper
        "FCM: The fuzzy c-means clustering algorithm" by James Bezdek
        Here we will use the Euclidean distance
        Pseudo code:
        1) Fix c, m, A
        c: n_centers
        m: 2 by default
        A: we are using Euclidean distance, so we don't need it actually
        2) compute the means (cluster centers)
        3) update the membership matrix
        4) compare the new membership with the old one, is difference is less than a threshold, stop. otherwise
            return to step 2)
    """

    def __init__(self, n_centers=2, m=2, max_iter=10000):
        self.n_centers = n_centers
        self.cluster_centers_: torch.Tensor = None
        self.u: torch.Tensor = None  # The membership
        self.m = m  # the fuzziness, m=1 is hard not fuzzy. see the paper for more info
        self.max_iter = max_iter

    def init_membership(self, num_of_points):
        self.init_membership_random(num_of_points)

    def init_membership_equal(self, num_of_points):
        """
        :param num_of_points:
        :return: nothing
        # In the below for loop, due to the rounding to 2 decimals, you may think that the membership sum for
        #  a point can be larger than 1. this can happen if number of clusters is larger than 10.
        # mathematical proof that this can happen:
        # (1) --- max_error per point membership to a single cluster is 0.01 (because of the rounding to 2 decimal
        #   points).
        # (2) --- (c-1) * 0.01 >= 1/c
        # (3) --- c^2 - c >= 1
        # solving for c we get c = 10.51 (approx.)
        # so when c >= 11, this error may occur.
        But I added a check below to prevent such a thing from happening
        """
        self.u = torch.zeros(num_of_points, self.n_centers)
        for i in torch.arange(num_of_points):
            row_sum = 0.0
            for c in torch.arange(self.n_centers):
                if c == self.n_centers-1:  # last iteration
                    self.u[i, c] = 1 - row_sum
                else:
                    rand_num = round(1.0/self.n_centers, 2)
                    if rand_num + row_sum >= 1.0:  # to prevent membership sum for a point to be larger than 1.0
                        if rand_num + row_sum - 0.01 >= 1.0:
                            self.logger.error('Something is not right in the init_membership')
                            return None
                        else:
                            self.u[i, c] = rand_num - 0.01
                    else:
                        self.u[i, c] = rand_num
                    row_sum += self.u[i, c]

    def init_membership_random(self, num_of_points):
        """
        :param num_of_points:
        :return: nothing
        """
        u_tmp = torch.rand(num_of_points, self.n_centers)
        u_tmp_sum = u_tmp.sum(1).unsqueeze(1).repeat(1, self.n_centers)
        u_tmp = u_tmp / u_tmp_sum
        self.u = u_tmp.double()

    def compute_cluster_centers(self, x: torch.Tensor):
        """
        :param x:
        :return:
        vi = (sum of membership for cluster i ^ m  * x ) / sum of membership for cluster i ^ m  : for each cluster i
        """
        n_fea = x.shape[1]
        for i in torch.arange(self.n_centers):
            u_i_exp = self.u[:, i].unsqueeze(1).repeat(1, n_fea).pow(self.m).double()
            smpl_u_mul = torch.mul(x, u_i_exp)
            smpl_u_sum = smpl_u_mul.sum(0)
            u_i_sum = u_i_exp.sum(0)
            center_i = smpl_u_sum / u_i_sum
            self.cluster_centers_[i, :] = center_i

    def update_membership(self, x):
        """
        update the membership matrix
        :param x: data points
        :return: nothing
        For performance, the distance can be computed once, before the loop instead of computing it every time
        """
        n_centers = self.cluster_centers_.shape[0]
        dist_mtrx = euclidean_distances(x, self.cluster_centers_)
        dist_mtrx = torch.tensor(dist_mtrx).double()
        for i in torch.arange(n_centers):
            dist_i_exp = dist_mtrx[:, i].unsqueeze(1).repeat(1, n_centers)
            dist_i_div = (dist_i_exp / dist_mtrx).pow(2.0 / (self.m - 1))
            dist_i_sum = dist_i_div.sum(1)
            u_i = torch.ones(1) / dist_i_sum
            self.u[:, i] = u_i

    def fit(self, x):
        """
        :param x:
        :return: self
        """
        if self.u is None:
            n_smpl = x.shape[0]
            self.init_membership_random(n_smpl)
        if self.cluster_centers_ is None:
            n_fea = x.shape[1]
            self.cluster_centers_ = torch.rand(self.n_centers, n_fea)
        theta = 0.000001
        center_loss = 1
        u_loss = 1
        while center_loss >= theta or u_loss >= theta:
            centers_old = self.cluster_centers_.clone()
            self.compute_cluster_centers(x)
            u_old = self.u.clone()
            self.update_membership(x)

            center_loss = torch.norm(self.cluster_centers_ - centers_old)
            u_loss = torch.norm(self.u - u_old)
        return self

    def predict(self, x):
        if self.u is None:
            u = None
        else:
            u = self.u.copy()
        self.u = torch.zeros(x.shape[0], self.n_centers)
        self.update_membership(x)
        predicted_u = self.u.clone()
        if torch.any(torch.isnan(predicted_u)):
            raise Exception("There is a nan in predict method")
        self.u = u
        return predicted_u
