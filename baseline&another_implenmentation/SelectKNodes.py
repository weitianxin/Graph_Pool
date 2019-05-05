import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt

def select_between_centrality(Gs,K=0):
    '''
    :param Gs: graph list
    :param K:  num of nodes we will select
    :return:   nodes we selected
    note: now, we select nodes only accordering to the pr val , may we need consider the best methods
    '''
    N=len(Gs)
    G=nx.Graph()
    TOPKs=[]
    KK = K
    for G in Gs:
        Ni = G.number_of_nodes()
        K = min(Ni,KK)
        bc_dict = nx.betweenness_centrality(G)
        bc = [(k, v) for k, v in bc_dict.items()]
        dtypes = [('idx',int),('bc_val', float)]
        bc=np.array(bc,dtype=dtypes)
        bc_order=np.sort(bc,order='bc_val')
        topK=[bc_order[-1-i]['idx'] for i in range(K)]
        TOPKs.append(topK)
    return TOPKs

def select_pagerank(Gs,K=0):
    '''
    :param Gs: graph list
    :param K: number of nodes we select
    :return: selected nodes
    :note: now, we select nodes only accordering to the pr val , may we need consider the best methods
    '''
    N=len(Gs)
    G=nx.Graph()
    TOPKs=[]
    max_Ni=0
    KK = K
    for G in Gs:
        Ni = G.number_of_nodes()
        K = min(KK, Ni)
        pr_dict = nx.pagerank(G)
        pr = [(k, v) for k, v in pr_dict.items()]
        dtypes = [('idx',int),('val', float)]
        pr = np.array(pr,dtype=dtypes)
        pr_order = np.sort(pr,order='val')
        topk_nodes = [pr_order[-1-i]['idx'] for i in range(K)]
        TOPKs.append(topk_nodes)
    return TOPKs