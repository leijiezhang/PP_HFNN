import torch


class NetworkTopology(object):
    def __init__(self, n, weights_type):
        # Constructor for NetworkTopology class
        self.N = n  # Number of nodes
        self.W = torch.zeros(n, n)  # Weights matrix
        self.A = torch.zeros(n, n)  # Adjacency matrix
        self.weights_type = weights_type  # Weights type (string)

    def get_neighbors(self, i):
        """
        Get index of neighbors of node i
        :param i:
        :return:
        """
        if i <= 0 or i > self.N:
            raise Exception('The node ID is not a valid value')
        else:
            idx = torch.nonzero(self.A[i, :])
        return idx

    def get_degree(self, i):
        """
        Get degree of node i
        :param i:
        :return:
        """
        if i is not None:
            if i <= 0 or i > self.N:
                raise Exception('The node ID is not a valid value')
            else:
                d = torch.sum(self.A[i, :])
        else:
            d = torch.sum(self.A, 1)
        return d

    def get_maxdegree(self):
        """
        Get the maximum degree
        :return:
        """
        max_d = torch.max(self.A, 1)
        return max_d

    def is_connected(self):
        """
        Check if the graph is connected
        :return:
        """
        