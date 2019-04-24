import numpy as np
import networkx as nx
import time

def Select_Tree(graphs,k,depth,max_nodes):
    pair_all = []
    for graph in  graphs:#every graph
        pg = nx.algorithms.link_analysis.pagerank_alg.pagerank_numpy(graph)
        pg_k = sorted(list(pg.items()),key=lambda x:x[1],reverse=True)[:k]
        pair_k = []
        for c in pg_k:# every tree
            node_id = c[0]
            node_all = []
            node_all.append(node_id)
            node_layer_old = list(graph[node_id])
            pair = []
            pair.extend([(node_id,node,1) for node in node_layer_old])
            node_all.extend(node_layer_old)
            for d in range(depth-1):
                node_layer_new = []
                for node_id in node_layer_old:
                    neighbor = list(graph[node_id])
                    new_node = [i for i in neighbor if i not in node_all]
                    pair.extend([(node_id,node,d+2) for node in new_node])
                    if new_node:
                        node_layer_new.extend(new_node)
                node_all.extend(node_layer_new)
                node_layer_old = node_layer_new
            pair_k.append(pair)
        pair_all.append(pair_k)
    return pair_all
