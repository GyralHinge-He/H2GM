import torch
import torch.nn as nn
import torch.nn.functional as F






class Econv(nn.Module):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """

    def __init__(self, in_features, out_features):
        super(Econv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        x = F.relu(torch.bmm(A, ax))  # has size (bs, N, num_outputs)

        return x


class Siamese_Econv(nn.Module):
    """
    Perform edge convolution on two input hypergraphs (g1, g2)
    """

    def __init__(self, in_features, num_features):
        super(Siamese_Econv, self).__init__()
        self.econv = Econv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.econv(*g1)
        emb2 = self.econv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2


class Hyperedge_to_node_conv(nn.Module):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """

    def __init__(self, in_features, out_features):
        super(Hyperedge_to_node_conv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A, x, norm=False):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        x = F.relu(torch.bmm(A, ax))  # has size (bs, N, num_outputs)

        return x
