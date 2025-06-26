import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class BitwiseMultipyLogis(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number,device):
        super(BitwiseMultipyLogis, self).__init__()
        self.features_num = features_num
        self.metapath_number = metapath_number
        self.dropout = dropout
        self.alpha = alpha
        self.leakyReLU = nn.LeakyReLU(alpha)
        self.logis = LogisticVector(features_num,1)
        self.device = device
        self.trans = nn.Parameter(torch.eye(features_num))
        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

    def forward(self, node_features, nodes_ori, layer_predict):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        node_features_temp = torch.zeros((2,node_features[0].size(0),node_features[0].size(1))).to(self.device)
        for kl in range(self.metapath_number):
            node_features_temp[kl] = torch.tanh(torch.matmul(node_features[kl],self.trans)+self.bias)
        node_features = node_features_temp
        layer_all_attention = torch.transpose(node_features, 0, 1) 
        semantic_pernode = self.layer_bitwise(layer_all_attention,layer_predict)  
        Z = torch.zeros(node_features.shape[1], node_features.shape[2]).cuda(self.device)
        adj_node = layer_all_attention 
        weight = self.logis(semantic_pernode)
        weight = F.softmax(weight, dim=1)
        for kk in range(self.metapath_number):
            if kk == layer_predict:
                continue
            Z = Z + weight[:,kk].unsqueeze(1) * adj_node[:,kk]
        X = node_features[layer_predict] + Z
        return X

    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1).cuda(self.device)
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        return b

    def layer_bitwise(self, node_layer_feature,layer_predict):
        layer_semantic = torch.zeros_like(node_layer_feature).cuda(self.device)
        exclude_dims = [i for i in range(self.metapath_number) if i != layer_predict]
        excluded_features = node_layer_feature[:, exclude_dims] * node_layer_feature[:, layer_predict].unsqueeze(1)
        self_features =  node_layer_feature[:,layer_predict] * node_layer_feature[:, layer_predict]
        layer_semantic[:, exclude_dims] = excluded_features
        layer_semantic[:, layer_predict] = self_features
        return layer_semantic


class LogisticVector(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden)
        self.active = nn.Sigmoid()
    def forward(self,x):
        value = self.parameter(x)
        out = self.active(value)
        return out.squeeze()