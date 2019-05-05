import torch
import torch.nn as nn
import  torch.nn.functional as F

class gnnMLP(nn.Module):
    def __init__(self,input_dim,hidden_dim=[64,64],out_dim=32):
        super(gnnMLP,self).__init__()
        mlp_layer = []
        pre_input_dim = input_dim
        for d in hidden_dim:
            mlp_layer.append(nn.Linear(pre_input_dim,d))
            mlp_layer.append(nn.BatchNorm1d(d))
            mlp_layer.append(nn.ReLU())
            mlp_layer.append(nn.Dropout())
            pre_input_dim = d
        mlp_layer.append(nn.Linear(pre_input_dim, out_dim))
        mlp_layer.append(nn.ReLU())
        self.mlp_model = nn.Sequential(*mlp_layer)

    def forward(self, h):
        outs = self.mlp_model(h)
        return outs

class GraphConv(nn.Module):
    def __init__(self,input_dim,hidden_dim=64, bias=True, use_mlp=False,mlp_hiddendim=[128],add_self=False):
        super(GraphConv,self).__init__()
        self.add_self = add_self
        self.weight = nn.Parameter(torch.FloatTensor(input_dim,hidden_dim))
        nn.init.xavier_uniform_(self.weight.data,gain=nn.init.calculate_gain('relu'))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
            nn.init.constant_(self.bias.data,1e-5)   # avoid loss gradient
        else:
            self.bias = None
        self.use_mlp = use_mlp
        if use_mlp:   #use mlp to chuli the features for one nodes
            self.mlp = gnnMLP(input_dim,hidden_dim=mlp_hiddendim,out_dim=hidden_dim)

    def forward(self, adj,h):
        if self.use_mlp:
            h = torch.matmul(adj, h)
            h = self.mlp(h)
        else:
            if self.add_self:
                h = h + torch.matmul(adj,h)           #add self
            else:
                h = torch.matmul(adj,h)
            h = torch.matmul(h,self.weight)
            if self.bias is not None:
                h = h + self.bias
            h = F.relu(h)
        return h

class GCNs(nn.Module):
    def __init__(self,input_dim,num_layers=2,hidden_dim=64,class_dim=2,pooling_type='sum',drop_rate=0.5,average_flag=0):
        super(GCNs, self).__init__()
        pre_dim = input_dim
        gcnlist = []
        self.drop_rate = drop_rate
        self.pooling = pooling_type
        self.avg_type = average_flag

        if self.pooling == 'attention':
            self.P = nn.Parameter(torch.FloatTensor(1,hidden_dim))
            nn.init.xavier_normal_(self.P.data,gain=nn.init.calculate_gain('relu'))
        for i in range(num_layers):
            gcnlist.append(GraphConv(pre_dim,hidden_dim=hidden_dim,use_mlp=False))
            pre_dim = hidden_dim
        self.gcnlist = nn.ModuleList(gcnlist)
        if self.pooling != 'others':       #not need pre
            self.pred = self.build_pre(hidden_dim,[128,32],class_dim,drop_rate=self.drop_rate)

    def build_pre(self, input_dim, hid_dim, class_dim,drop_rate=0.5):  # now without BN layer
        pre_input_dim = input_dim
        pre_layer = []
        for pre_dim in hid_dim:
            pre_layer.append(nn.Linear(pre_input_dim, pre_dim))
            pre_layer.append(nn.BatchNorm1d(pre_dim))
            pre_layer.append(nn.ReLU())
            pre_layer.append(nn.Dropout(p=drop_rate))
            #pre_layer.append(nn.BatchNorm1d(pre_dim))
            pre_input_dim = pre_dim
        pre_layer.append(nn.Linear(pre_input_dim, class_dim))
        #pre_layer.append(nn.ReLU())           # the last layer before softmax, don't need activate.
        pred_model = nn.Sequential(*pre_layer)
        return pred_model

    def forward(self,adj,h,number_nodes=None):
        for layer in self.gcnlist:
            h = layer(adj,h)
        if self.pooling == 'sum':
            h = torch.sum(h,dim=-2)
        elif self.pooling == 'average':  # here really contain the information : contains how many None nodes.
            if self.avg_type==0:   #in some how,this equtal to sum. and this can be good!!
                h = torch.sum(h,dim=-2)
                num_nodes = 1/torch.sum(number_nodes,dim=-1)
                num = torch.diagflat(num_nodes)
                h = torch.matmul(num, h)
            else:   #this real average
                num_nodes = 1 / torch.sum(number_nodes, dim=-1)
                mask = torch.matmul(num_nodes.unsqueeze(-1),number_nodes)
                h = torch.matmul(mask,h).squeeze(-2)


        elif self.pooling == 'max':
            h,_ = torch.max(h,dim=-2)
        elif self.pooling == 'attention':
            e = torch.matmul(self.P,torch.transpose(h,-1,-2))
            e = F.relu(e)
            e1 = -9e15 * torch.ones_like(e)
            e = torch.where(number_nodes>=0, e, e1)
            e = F.softmax(e,dim=-1)
            h = torch.matmul(e,h)
            h = torch.squeeze(h,-2)
        else:        # this just a feature exacture ,and don't need prediction
            return h   # as the input of next_lyer

        pred = self.pred(h)
        #pred = F.softmax(pred,dim=-1)

        return pred

    def loss(self,pre,label):
        pre = pre.squeeze(-2)
        return F.cross_entropy(pre,label)









