import networkx as nx
import numpy as np
import torch
import torch.utils.data

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list,G_trees, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.adj_tree_all = []
        self.c_nodes = []
        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        for G_tree in G_trees:
            #对角线是负1
            adj_tree_padded = -1*np.eye(self.max_num_nodes)
            c_node = []
            for tree in G_tree:
                c_node.append(tree[0][0])
                for i,j,d in tree:
                    if d>=np.max(adj_tree_padded[i]):
                        adj_tree_padded[i,j] = d
            self.c_nodes.append(c_node)
            self.adj_tree_all.append(adj_tree_padded)
        self.c_nodes = np.array(self.c_nodes)
        #if features == 'default':
        # using node labels as feature
        self.feat_dim = G_list[0].node[0]['feat'].shape[0] #default:dim of node-label

        for index,G in enumerate(G_list):
            adj = np.array(nx.to_numpy_matrix(G))#不带自连通
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)#D^-1/2AD^-1/2
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                # using node labels as feature
                f[i,:] = G.node[u]['feat']
            self.feature_all.append(f)


        self.feat_dim = self.feature_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        c_nodes = self.c_nodes[idx]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        adj_tree_padded = self.adj_tree_all[idx]
        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                "c_nodes":c_nodes,
                "adj_tree":adj_tree_padded,
                'feats':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes}

