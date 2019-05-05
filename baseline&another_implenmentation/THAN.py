import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import GraphCnn as GC

'''
now,we can also use mask to form the network.
band use the adj matrix to present the neighthod but may be low efficent for sparse matrix
'''

class THAN_one_layer(nn.Module):  #use adj get child derectly
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
        e = torch.where(adj != 0, e, pos0)
        attention = F.softmax(e, dim=-1)
        embedding = torch.matmul(attention,h)

        ones = torch.ones((B,1,d)).cuda()

        h_root = torch.unsqueeze(h_root,dim=-2)
        assign = torch.matmul(h_root.transpose(-1, -2),ones)

        embedding = torch.where(assign>0, embedding, h0)

        return embedding




class THAN_ADJ(nn.Module):   #use adj get child derectly
    def __init__(self,input_feature_dim=80,hidden_dim=64,max_depth=4,need_feature_chang=True, pre_didden_dim=[128,32],num_calss=2,dropout=0):
        super(THAN_ADJ,self).__init__()
        self.hidden_dim=hidden_dim
        self.Depth=max_depth
        self.droput=dropout
        self.need_feature_change = need_feature_chang
        if self.need_feature_change:
            self.weight = nn.Parameter(torch.FloatTensor(input_feature_dim,hidden_dim))
            nn.init.xavier_uniform_(self.weight.data,gain=nn.init.calculate_gain('relu'))
        self.P = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.P.data,gain=nn.init.calculate_gain('relu'))
        attention_list = []
        for i in range(self.Depth):
            attention_list.append(THAN_one_layer(hidden_dim=64))
        self.attention_list = nn.ModuleList(attention_list)
        self.pre = self.build_pre(self.hidden_dim, pre_didden_dim, num_calss, dropout = self.droput)

    def build_pre(self,input_dim,hid_dim,class_dim,dropout=0):  #now without BN layer
        pre_input_dim = input_dim
        pre_layer = []
        for pre_dim in hid_dim:
            pre_layer.append(nn.Linear(pre_input_dim,pre_dim))
            pre_layer.append(nn.BatchNorm1d(pre_dim))
            pre_layer.append(nn.ReLU())
            pre_layer.append(nn.Dropout(p=dropout))
            pre_input_dim = pre_dim
        pre_layer.append(nn.Linear(pre_input_dim, class_dim))
        pre_layer.append(nn.ReLU())
        pred_model = nn.Sequential(*pre_layer)
        return pred_model

    def forward(self, adj,X,root_list):
        if self.need_feature_change:    #featuew change
            h = torch.matmul(X,self.weight)
            h = F.relu(h)
        else:
            h = X

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
        pre = torch.squeeze(pre, dim=-1)
        pre = F.softmax(pre,dim=-1)
        pre = torch.squeeze(pre, -2)
        return pre

    def loss(self, pre, label):
        return F.cross_entropy(pre, label)



class TreeGCN(nn.Module): # a completed model ,gcn + tree model
    def __init__(self, input_dim, attention=False, num_layers=2, hidden_dim=64, class_dim=2,pre_hidden_dim=[128], pooling_type='others', drop_rate=0, max_depth=4, need_feature_chang=False):
        super(TreeGCN,self).__init__()
        self.gcn_model = GC.GCNs(input_dim,num_layers=num_layers,hidden_dim=hidden_dim,class_dim=0,\
                                 pooling_type=pooling_type,drop_rate=drop_rate)

        #self.bn = nn.BatchNorm2d()
        self.attention_flag = attention
        if self.attention_flag:  # Tree attention
            print("Use Tree attention Pooling methods!not directly use adj!!")
            self.THAN_model = TreeAttention(input_feature_dim=hidden_dim, hidden_dim=hidden_dim, max_depth=max_depth,\
                                   need_feature_chang=need_feature_chang, num_calss=class_dim, pre_didden_dim=pre_hidden_dim,dropout=drop_rate)
        else:                   # Tree average
            print("Use Tree average Pooling methods!")
            self.TAver_model = TreeAverage(hidden_dim,max_depth=max_depth,prehidden_dim=pre_hidden_dim,class_dim=class_dim,dropout=drop_rate)

    def aply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, adj, h0, rootlist, num_nodes=None):
        h = self.gcn_model(adj, h0)
        #h = self.aply_bn(h)
        if self.attention_flag:
            pre = self.THAN_model(adj, h, rootlist)
        else:
            pre = self.TAver_model(adj, h, rootlist)
        pre = torch.squeeze(pre, -2)
        return pre

    def loss(self, pre, label):
        return F.cross_entropy(pre, label)


class TreeAverage(nn.Module):
    def __init__(self,input_dim, out_dim=0, max_depth=4, prehidden_dim=[], class_dim=2,dropout=0,trans_flag=False):
        super(TreeAverage, self).__init__()
        if out_dim == 0:
            out_dim = input_dim
        self.max_depth = max_depth
        self.trans_flag = trans_flag
        if self.trans_flag:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, out_dim))
            nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        self.pred_module = self.build_pre(out_dim, prehidden_dim, class_dim,dropout=dropout)

    def avg_forward(self, h, adj, h_root, h_child):
        if self.trans_flag:
            h = torch.matmul(h,self.weight)
            h = F.relu(h)
        N = adj.shape[1]
        B = h.shape[0]
        ones_vec = torch.ones_like(adj)
        zeors_vec = torch.zeros_like(adj)
        adj_new = torch.where(adj != 0, ones_vec, zeors_vec)
        assign_tree = torch.matmul(h_root.unsqueeze(-1), h_child.unsqueeze(-2))
        adj_new = torch.mul(adj_new, assign_tree)
        eyes = torch.eye(N).repeat(B, 1, 1).cuda()
        adj_new = torch.where(eyes == 1, eyes, adj_new)  # 对角线置为1

        #normlized!
        adj_degree = 1.0 / torch.sum(adj_new, dim=-1)
        adj_degree = torch.unsqueeze(adj_degree, dim=-1)
        adj_degree = torch.matmul(adj_degree, torch.ones_like(adj_degree).transpose(-1, -2))
        adj_new = torch.mul(adj_degree,adj_new)
        h = torch.matmul(adj_new, h)
        return h

    def build_pre(self,input_dim,hid_dim,class_dim,dropout=0):  #now without BN layer
        pre_input_dim = input_dim
        pre_layer = []
        for pre_dim in hid_dim:
            pre_layer.append(nn.Linear(pre_input_dim,pre_dim))
            pre_layer.append(nn.BatchNorm1d(pre_dim))
            pre_layer.append(nn.ReLU())
            pre_layer.append(nn.Dropout(p=dropout))
            pre_input_dim = pre_dim
        pre_layer.append(nn.Linear(pre_input_dim, class_dim))
        pre_layer.append(nn.ReLU())
        pred_model = nn.Sequential(*pre_layer)
        return pred_model

    def forward(self, adj, x, tree_info):
        #print(x.shpae)
        if self.trans_flag:
            h = torch.matmul(x,self.weight)
            h = F.relu(h)

        else:
            h = x
        for i in range(self.max_depth-1):
            h_root = tree_info[:, self.max_depth-2-i, :]
            h_child = tree_info[:, self.max_depth-1-i, :]
            h = self.avg_forward(h, adj, h_root, h_child)
        h_root = tree_info[:, 0, :]
        h_root = h_root.unsqueeze(-2)
        # h_root_sum =1.0 / torch.sum(h_root,dim=-1).unsqueeze(-1)   # if average  ,not this code represet is sum!!
        # h_root = torch.matmul(h_root_sum,h_root)
        out = torch.matmul(h_root, h).squeeze(-2)  #root node,the number of nodes==tree's num
        pre = self.pred_module(out).squeeze(-2)
        #pre = F.softmax(pre,dim=-1)
        return  pre

    def loss(self,pre,label):
        return F.cross_entropy(pre,label)



class TreeAttention_onelayer(nn.Module):  #not use adj get child directly, ie the child not repeat in a tree
    def __init__(self,hidden_dim=64):
        super(TreeAttention_onelayer, self).__init__()

        self.relu = F.relu
        self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))

        self.a1 = nn.Parameter(torch.FloatTensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.a1.data,gain=nn.init.calculate_gain('relu'))
        self.a2 = nn.Parameter(torch.FloatTensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.a2.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h, adj, h_root,h_child):
        h0 = h
        N = h.shape[1]
        B = h.shape[0]
        d = h.shape[-1]

        ones_vec = torch.ones_like(adj)
        zeors_vec = torch.zeros_like(adj)
        adj_new = torch.where(adj != 0, ones_vec, zeors_vec)
        assign_tree = torch.matmul(h_root.unsqueeze(-1), h_child.unsqueeze(-2))
        adj_new = torch.mul(adj_new, assign_tree)
        eyes = torch.eye(N).repeat(B, 1, 1).cuda()
        adj_new = torch.where(eyes == 1, eyes, adj_new)  # 对角线置为1
        adj = adj_new              # get the new adj , not allow the repeat node in a tree

        h = torch.matmul(h,self.weight)
        h = self.relu(h)
        t1 = torch.matmul(h,self.a1)
        t2 = torch.matmul(h,self.a2)

        e0 = t1.repeat(1, 1, N).view(B, N * N, -1) + t2.repeat(1, N, 1)
        e = e0.view(B, N, N)
        e = self.relu(e)

        pos0 = -9e15 * ones_vec
        e = torch.where(adj != 0, e, pos0)
        attention = F.softmax(e, dim=-1)
        embedding = torch.matmul(attention,h)
        return embedding

class TreeAttention(nn.Module):  #tree attention, node not repeat in a tree
    def __init__(self,input_feature_dim=80,hidden_dim=64,max_depth=4,need_feature_chang=False, pre_didden_dim=[128,32],num_calss=2,dropout=0):
        super(TreeAttention,self).__init__()
        self.hidden_dim = hidden_dim
        self.Depth = max_depth
        self.droput = dropout
        self.need_feature_change = need_feature_chang
        self.readout = 'attention'
        if self.need_feature_change:
            self.weight = nn.Parameter(torch.FloatTensor(input_feature_dim,hidden_dim))
            nn.init.xavier_uniform_(self.weight.data,gain=nn.init.calculate_gain('relu'))
        self.P = nn.Parameter(torch.FloatTensor(hidden_dim,1))
        nn.init.xavier_uniform_(self.P.data,gain=nn.init.calculate_gain('relu'))
        attention_list = []
        for i in range(self.Depth):
            attention_list.append(TreeAttention_onelayer(hidden_dim=64))
        self.attention_list = nn.ModuleList(attention_list)
        self.pre = self.build_pre(self.hidden_dim, pre_didden_dim, num_calss, dropout = self.droput)

    def build_pre(self,input_dim,hid_dim,class_dim,dropout=0):  #now without BN layer
        pre_input_dim = input_dim
        pre_layer = []
        for pre_dim in hid_dim:
            pre_layer.append(nn.Linear(pre_input_dim,pre_dim))
            pre_layer.append(nn.BatchNorm1d(pre_dim))
            pre_layer.append(nn.ReLU())
            pre_layer.append(nn.Dropout(p=dropout))
            pre_input_dim = pre_dim
        pre_layer.append(nn.Linear(pre_input_dim, class_dim))
        pre_layer.append(nn.ReLU())
        pred_model = nn.Sequential(*pre_layer)
        return pred_model

    def forward(self, adj,X,root_list):
        if self.need_feature_change:    #featuew change
            h = torch.matmul(X,self.weight)
            h = F.relu(h)
        else:
            h = X

        for i in range(self.Depth-2):
            h_root = root_list[:,self.Depth-2-i,:]  #提取第i个depth的root
            h_child = root_list[:, self.Depth - 1 - i, :]  # 提取第i个depth的root
            h = self.attention_list[i](h,adj,h_root,h_child)

        if self.readout == 'attention': #if use attention to get the graph representation
            e = torch.matmul(h,self.P).squeeze(-1)
            zeros_vec = torch.ones_like(h_root)
            e = torch.where(h_root !=0, e, zeros_vec)
            e = F.relu(e)
            attention = F.softmax(e)
            h_root = attention
        h = torch.matmul(h_root.unsqueeze(-2),h).squeeze(-2)  #only use sum to get graph
        pre = self.pre(h)
        return pre


    def loss(self, pre, label):
        return F.cross_entropy(pre, label)

























                    




















