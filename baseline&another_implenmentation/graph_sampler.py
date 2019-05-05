import networkx as nx
import numpy as np
import torch
import torch.utils.data

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=True,  max_num_nodes=1000):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        self.feat_dim = len(G_list[0].node[0]['feat'])

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    if i>= self.max_num_nodes:
                       break
                    f[i,:] = G.node[u]['feat']
                self.feature_all.append(f)
        self.feat_dim = self.feature_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = min(adj.shape[0],self.max_num_nodes)
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        if num_nodes<self.max_num_nodes:
           adj_padded[:num_nodes, :num_nodes] = adj
        else:
           adj_padded[:,:] =adj[:self.max_num_nodes,:self.max_num_nodes]

        num = np.zeros((1,self.max_num_nodes))
        num[:,:num_nodes] = 1
        num_nodes = num


        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                'feature':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes
                }

