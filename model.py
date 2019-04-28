import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)).cuda())
        #change********
        self.a1 = nn.Parameter(torch.zeros(size=(out_features,1)).cuda())
        self.a2 = nn.Parameter(torch.zeros(size=(out_features,1)).cuda())
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, adj_tree,d1,d2=-1):
        h = torch.matmul(input, self.W)
        N = h.size()[1]#max_node_size
        size = h.size()[0]
        t1 = torch.matmul(h,self.a1)
        t2 = torch.matmul(h,self.a2)

        e0 = t1.repeat(1, 1, N).view(size, N*N, -1) + t2.repeat(1, N, 1)
        e = e0.view(size, N, N)
        e = self.leakyrelu(e)
        # a_input = torch.cat([h.repeat(1, 1, N).view(size,N * N, -1), h.repeat(1, N, 1)], dim=1)
        # a_input = a_input.view(size, N, N, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj == d1 , e, zero_vec)
        attention = torch.where(adj==d2, e, attention)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return h_prime
 

class GraphTreeAverageLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphTreeAverageLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('leaky_relu'))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, adj_tree, d1, d2=-1):
        h = torch.matmul(input, self.W)
        h = F.relu(h)
        one_vec = torch.ones_like(adj)
        zero_vec = torch.ones_like(adj)
        adj_new = torch.where(adj == d1, one_vec, zero_vec)
        adj_new = torch.where(adj == d2, one_vec, adj_new)
        adj_degree =1.0 / torch.sum(adj_new,dim=-1)
        adj_degree = torch.unsqueeze(adj_degree,dim=-1)
        adj_degree = torch.matmul(adj_degree,torch.ones_like(adj_degree).transpose(-1,-2))
        adj_new = torch.mul(adj_new,adj_degree)   #change****
        h_prime = torch.matmul(adj_new,h)
        return h_prime
    
class gcn_tree(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,depth,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, alpha=0.2,args=None):
        super(gcn_tree, self).__init__()
        self.depth = depth
        add_self = not concat
        self.bias = args.bias
        self.bn = bn
        self.concat = concat
        self.num_layers = num_layers
        self.label_dim = label_dim
        self.act = nn.ReLU()
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim,hidden_dim,
                embedding_dim,num_layers,add_self,normalize=False,dropout=dropout)
        if arg.attention:
            self.hi_att = nn.ModuleList([GraphAttentionLayer(embedding_dim,embedding_dim,dropout,alpha,concat)
                                     for _ in range(depth)])
        else:
            self.hi_att = nn.ModuleList([GraphTreeAverageLayer(embedding_dim,embedding_dim,dropout,alpha,concat)
                                     for _ in range(depth)])
            

        self.pred_layers = self.build_pred_layers(embedding_dim,pred_hidden_dims,label_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def forward(self,X,adj,adj_tree,c_nodes,batch_num_nodes):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        embedding_tensor = self.gcn_forward(X,adj,self.conv_first,self.conv_block,self.conv_last,
                                            concat=self.concat,embedding_mask=embedding_mask)
        for i in range(self.depth):
            embedding_tensor = self.hi_att[i](embedding_tensor,adj,adj_tree,d1=self.depth-i)
        tensor_list = []
        for i,node_index in enumerate(c_nodes):
            temp = embedding_tensor[i, node_index, :]
            tensor_list.append(temp)
        embedding_tensor = torch.stack(tensor_list,dim=0)

        final_tensor = torch.sum(embedding_tensor,dim=1)#需要改
        output = self.pred_layers(final_tensor)
        return output

    def loss(self,pred,label):
        return F.cross_entropy(pred, label, size_average=True)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias)
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None,concat=True):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        if concat:
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor
