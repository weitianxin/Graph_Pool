'''
in this file we creat the trees accordering the selected nodes
'''
import networkx as nx
import numpy as np
import torch
import SelectKNodes as select
import torch.utils.data


class Gtree(object):
    def __init__(self, rt, father=None):
        self.rt = rt
        self.childs = None
        self.father = father
        self.feature = None
        self.depth = 0

def creat_onetree(rt,adj_list,node_flag,depth=0,father=None,max_depth=3):
    '''
    :param rt: root nodes
    :param adj_list:  neighborhods
    :param depth: depth
    :param father: father nodes
    :param max_depth: the max K hope will be used
    :return: one tree or a subtree
     we may need other code to judge weather circle,
    '''
    node_flag[rt] += 1
    gt = Gtree(rt, father=father)
    gt.depth = depth
    gt.father = father

    if len(adj_list[rt]) == 0:
        return gt

    gt.childs = []
    if depth == max_depth-1: #最后一层不需要childrens
        return gt
    for child in adj_list[rt]:
        if node_flag[child] == 0:   #if the node have not existed in this tree.
            gt.childs.append(creat_onetree(child, adj_list,node_flag, depth=depth+1, father=rt,max_depth=max_depth))
    return gt


def CreatTree_forOneGraph(root_nodes,G:nx.Graph,add_super_nodes=False,max_depth=3):
    '''
    creat trees for one graph
    :param G:
    :param root_nodes:
    :return:  k tree for the graph
    '''
    # here we can use a super nodes to conect all the trees for one graph
    gtrees=[]
    N = G.number_of_nodes()
    #node_flag = np.zeros((N))
    for rt in root_nodes:
        node_flag = np.zeros((N))
        #adj_list=G.adjacency_list()
        adj = G.adj
        adj_list = []
        for key, val in adj.items():
            nei = [k for k in val.keys()]
            adj_list.append(nei)
        one_tree = creat_onetree(rt, adj_list, node_flag, depth= 0, max_depth= max_depth)
        gtrees.append(one_tree)
    if add_super_nodes:
        return
        pass
    return gtrees

def BFSread_oneTree(tree:Gtree):
    '''
    read the tree and return list
    :param tree:
    :return:
    '''
    query = [tree]
    data = {0: tree.rt }
    depth = tree.depth
    childs = tree.childs
    query.extend(childs)
    while len(query)>0:
        now_node=query[0]
        if now_node.depth != depth :
            data[depth] = childs
            depth = now_node.depth
            childs = []
            del query[0]
        child=now_node.childs
        childs.append(child)
        query.extend(child)

    return data





def BFS_read_OneGraph(gtrees,K,N,max_depth):# for one graph
    """
    :param gtrees: trees of one graph
    :param K:    the number of trees
    :param N:    the total number of nodes
    :param max_depth:  max depth of all trees
    :return: data like array[max_depth,K,N]
    """
    data = np.zeros([max_depth,K,N],dtype=np.float)
    k=0
    for tr in gtrees:
        query =[tr]
        depth = 0
        while len(query)>0:
            now_node = query[0]
            if depth!=now_node.depth:
                depth += 1
            try:
                query.extend(now_node.childs)
            except:
                pass
            idx =int(now_node.rt)
            data[depth,k,idx] = 1
            del query[0]
        k += 1
    return data



def BFS_read_OneGraph2(gtrees,K,N,max_depth):# for one graph
    """
    :param gtrees: trees of one graph
    :param K:    the number of trees
    :param N:    the total number of nodes
    :param max_depth:  max depth of all trees
    :return: data like array[max_depth,N]
    """
    data = np.zeros([max_depth, N], dtype=np.float)
    for tr in gtrees:
        query = [tr]
        depth = 0
        while len(query)> 0:
            now_node = query[0]
            if depth != now_node.depth:
                depth += 1
            try:
                query.extend(now_node.childs)
            except:
                pass
            idx =int(now_node.rt)
            data[depth, idx] = 1
            del query[0]
    return data

def BFS_read_OneGraph3(gtrees,K,N,max_depth):# for one graph
    """
    not allow that the same node repeat in the same tree
    :param gtrees: trees of one graph
    :param K:    the number of trees
    :param N:    the total number of nodes
    :param max_depth:  max depth of all trees
    :return: data like array[max_depth,N]
    """
    data = np.zeros([max_depth, N], dtype=np.float)

    for tr in gtrees:
        node_flag = np.zeros([1, N], dtype=np.int)  #之前是 depth*N
        query = [tr]
        depth = 0
        while len(query) > 0:
            now_node = query[0]
            if depth != now_node.depth:
                depth += 1
            idx =int(now_node.rt)
            if node_flag[ 0, idx] == 0:   # here we avoid that the same node exist many time in a tree
                data[depth, idx] = 1
                node_flag[0, idx] += 1
                try:
                    query.extend(now_node.childs)
                except:
                    pass
            del query[0]
    return data

def get_graphs_tree_data(Gs,max_depth,K=5):
    '''
    get the tree information , for each depth return a mask tell if node exists
    :param Gs:graphs
    :param max_depth: max depth
    :param K: the number of trees for one graph
    :return:  tree information, each depth layer contain which nodes ,use nodes mask to represent them
    '''
    topk_nodes = select.select_pagerank(Gs,K)
    #topk_nodes = select.select_between_centrality(Gs,K)
    tree_info = []
    for i in range(len(Gs)):
        root_nodes = topk_nodes[i]
        g = Gs[i]
        gtrees = CreatTree_forOneGraph(root_nodes,g,max_depth=max_depth)
        data = BFS_read_OneGraph2(gtrees,K,g.number_of_nodes(),max_depth=max_depth)
        tree_info.append(data)
    return tree_info



class graph_sampler(torch.utils.data.Dataset):
    '''
    dataset for graph
    '''
    def __init__(self,Gs,max_depth=10,max_node=1000,tree_num=8,normalize=True):
        tree_info = get_graphs_tree_data(Gs,max_depth,K=tree_num)
        self.adj_all = []
        self.feature_all = []
        self.tree_info = tree_info
        self.label_all = []
        self.max_node = max_node

        self.feature_dim = len(Gs[0].node[0]['feat'])
        for g in Gs:
            adj = nx.adj_matrix(g).todense()
            adj = np.array(adj)
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

            f = np.zeros([self.max_node, self.feature_dim], dtype=np.float)
            for i, u in enumerate(g.nodes()):
                if i < self.max_node:
                    f[i, :] = np.array(g.node[u]['feat'])
                else:
                    break
            self.feature_all.append(f)
            self.adj_all.append(adj)
            self.label_all.append(g.graph['label'])

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_node, self.max_node))
        if num_nodes<self.max_node:
            adj_padded[:num_nodes, :num_nodes] = adj
        else:
            adj_padded[:self.max_node,:self.max_node] = adj[:self.max_node,:self.max_node]
        tree_info = self.tree_info[idx]
        tree_info_shape = np.array(tree_info.shape)
        tree_info_shape[-1] = self.max_node
        tree_info_padded = np.zeros(tuple(tree_info_shape))
        if num_nodes<self.max_node:
            tree_info_padded[:,:num_nodes] = tree_info
        else:
            tree_info_padded[:,:self.max_node] = tree_info[:,:self.max_node]

        return {'adj':adj_padded,
                'number_nodes':num_nodes,
                'label':self.label_all[idx].copy(),
                'feature': self.feature_all[idx].copy(),
                'tree_info':tree_info_padded
                }


