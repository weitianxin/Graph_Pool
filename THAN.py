import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import GraphCnn as GC

'''
now,we can also use mask to form the network.
band use the adj matrix to present the neighthod but may be low efficent for sparse matrix
'''

class THAN_one_layer(nn.Module):
    def __init__(self,hidden_dim=64):
        super(THAN_one_layer, self).__init__()

        self.relu = F.relu
        self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))

        self.a1 = nn.Parameter(torch.FloatTensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.a1.data,gain=nn.init.calculate_gain('relu'))
        self.a2 = nn.Parameter(torch.FloatTensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.a2.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h, adj, h_root):
        h0 = h
        N = h.shape[1]
        B = h.shape[0]
        d = h.shape[-1]

        h = torch.matmul(h,self.weight)
        h = self.relu(h)
        t1 = torch.matmul(h,self.a1)
        t2 = torch.matmul(h,self.a2)

        e0 = t1.repeat(1, 1, N).view(B, N * N, -1) + t2.repeat(1, N, 1)
        e = e0.view(B, N, N)
        e = self.relu(e)

        pos0 = -9e15 * torch.zeros_like(adj)
        e = torch.where(adj> 0, e, pos0)
        attention = F.softmax(e, dim=-1)
        embedding = torch.matmul(attention,h)

        ones = torch.ones((B,1,d)).cuda()

        h_root = torch.unsqueeze(h_root,dim=-2)
        assign = torch.matmul(h_root.transpose(-1, -2),ones)

        embedding = torch.where(assign>0, embedding, h0)

        return embedding

class THAN_ADJ(nn.Module):
    def __init__(self,input_feature_dim=80,hidden_dim=64,max_depth=4,need_feature_chang=True):

        super(THAN_ADJ,self).__init__()
        self.hidden_dim=hidden_dim
        self.Depth=max_depth
        self.need_feature_change = need_feature_chang
        if self.need_feature_change:
            self.weight = nn.Parameter(torch.FloatTensor(input_feature_dim,hidden_dim))
            nn.init.xavier_uniform_(self.weight.data,gain=nn.init.calculate_gain('leaky_relu'))
        self.P = nn.Parameter(torch.FloatTensor(1,hidden_dim))
        nn.init.xavier_uniform_(self.P.data,gain=nn.init.calculate_gain('relu'))
        attention_list = []
        for i in range(self.Depth):
            attention_list.append(THAN_one_layer(hidden_dim=64))
        self.attention_list = nn.ModuleList(attention_list)
        self.pre = self.build_pre(self.hidden_dim,[128,64],2)

    def build_pre(self,input_dim,hid_dim,class_dim):  #now without BN layer
        pre_input_dim = input_dim
        pre_layer = []
        for pre_dim in hid_dim:
            pre_layer.append(nn.Linear(pre_input_dim,pre_dim))
            pre_layer.append(nn.BatchNorm1d(pre_dim))
            pre_layer.append(nn.ReLU())
            pre_layer.append(nn.Dropout(p=0.5))
            pre_input_dim = pre_dim
        pre_layer.append(nn.Linear(pre_input_dim, class_dim))
        pre_layer.append(nn.ReLU())
        pred_model = nn.Sequential(*pre_layer)
        return  pred_model

    def forward(self, adj,X,root_list):
        if self.need_feature_change:    #featuew change
            h=torch.matmul(X,self.weight)
            h=F.relu(h)
        else:
            h =X

        for i in range(self.Depth-1):
            root = root_list[:,self.Depth-1-i,:]  #提取第i个depth的root
            h = self.attention_list[i](h,adj,root)

        e = torch.matmul(self.P,torch.transpose(h, -1, -2)).squeeze(-2)
        e = F.relu(e)
        zeros_vec = -9e15*torch.ones_like(root)
        e = torch.where(root>0, e, zeros_vec)
        attention = F.softmax(e, dim=-1)
        out = torch.matmul(attention.unsqueeze(-2), h).squeeze(-2)
        pre = self.pre(out)
        pre = torch.squeeze(pre,dim=-1)
        pre = F.softmax(pre,dim=-1)
        pre = torch.squeeze(pre, -2)
        return pre

    def loss(self,pre,label):
        return F.cross_entropy(pre,label)



class THANaddGCN(nn.Module):
    def __init__(self,input_dim,num_layers=2,hidden_dim=64,class_dim=2,pooling_type='others',drop_rate=0.5,max_depth=4,need_feature_chang=False):
        super(THANaddGCN,self).__init__()
        self.gcn_model = GC.GCNs(input_dim,num_layers=num_layers,hidden_dim=hidden_dim,class_dim=0,\
                                 pooling_type=pooling_type,drop_rate=drop_rate)

        #self.bn = nn.BatchNorm2d()
        self.THAN_model = THAN_ADJ(input_feature_dim=hidden_dim,hidden_dim=hidden_dim,max_depth=max_depth,need_feature_chang=need_feature_chang)

    def aply_bn(self,x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, adj,h0,rootlist,num_nodes=None):
        h = self.gcn_model(adj,h0)
        h = self.aply_bn(h)
        pre = self.THAN_model(adj,h,rootlist)
        pre = torch.squeeze(pre,-2)
        return pre

    def loss(self,pre,label):
        return F.cross_entropy(pre,label)




























                    




















