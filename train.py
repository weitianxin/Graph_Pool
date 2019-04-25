from sampler import GraphSampler
import argparse
import load_data
from select_tree import Select_Tree
import torch
from parse_my import arg_parse
import numpy as np
from model import gcn_tree
from torch.autograd import Variable
import time
from torch import nn
import sklearn.metrics as metrics
import random
from tensorboardX import SummaryWriter
import os
import time

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):

    model.eval()
    labels = []
    preds = []
    avg_loss = 0
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        c_nodes = data['c_nodes']
        adj_tree = Variable(data["adj_tree"], requires_grad=False).cuda()
        label_ = Variable(data['label'].long(), requires_grad=False).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()

        ypred = model(h0, adj, adj_tree, c_nodes, batch_num_nodes)
        loss = model.loss(ypred,label_)
        avg_loss += float(loss)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    avg_loss /= (batch_idx+1)

    result = {
              'acc': metrics.accuracy_score(labels, preds),
              'loss': avg_loss
             }
    return result

def train(train_dataset, model, args, val_dataset, out_file="val.txt",test_dataset=None,writer=None):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=0.0001)
    iter = 0
    train_accs = []
    train_epochs = []

    val_accs = []
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        #print('Epoch: ', epoch)
        for batch_idx, data in enumerate(train_dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy()
            c_nodes = data['c_nodes']
            adj_tree = Variable(data["adj_tree"],requires_grad=False).cuda()
            ypred = model(h0, adj, adj_tree,c_nodes,batch_num_nodes)  # batch_num_nodes每个图有几个结点
            loss = model.loss(ypred, label)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        #print('Avg loss: ', avg_loss, '; epoch time: ', elapsed)
        result = evaluate(train_dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
            # print("epoch:{:3d},..time:{:.2f},...avg_loss:{:.3f},...,train acc:{:.3f},\
            #  ...val acc:{:.3f}".format(epoch, elapsed, avg_loss, result['acc'], val_result['acc']))

            # # new!
            # with open(out_file, "a+") as f:
            #     f.write("epoch: " + str(epoch) + " " + str(result['acc']) + " " + str(val_result['acc']) + "\n")

        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='test')
            print("epoch:{:3d},..time:{:.2f},...avg_loss:{:.3f},...,train acc:{:.3f},...val acc:{:.3f},\
            ,..test acc:{:.3f}".format(epoch, elapsed, avg_loss, result['acc'], val_result['acc'],test_result['acc']))

        if writer != None:  #记录训练日志
            writer.add_scalars('scalar/acc', {'train': result['acc'], 'val': val_result['acc']}, epoch)
            loss_nol2 = avg_loss
            if test_dataset is not  None:
                writer.add_scalars('scalat/ce_loss', {'train_loss': result['loss'], 'val_loss': val_result['loss']}, epoch)
            else:
                writer.add_scalars('scalat/ce_loss', {'train_loss':result['loss'],'val_loss':val_result['loss']}, epoch)
            writer.add_scalars('scalat/train+ce+L2_loss',{'model_loss': loss_nol2, 'L2 loss': 0,\
                                                    'ce_loss':avg_loss }, epoch)


if __name__=="__main__":
    args = arg_parse()
    writer_path = args.method + "-lr" + str(args.lr)+'drop'+str(args.dropout)
    print(writer_path)
    try:  # 日志文件
        os.removedirs(writer_path)
    except:
        pass
    writer = SummaryWriter(log_dir=writer_path)

    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    print("read file over!!")
    #using node labels as feature
    for G in graphs:
        for u in G.nodes():
            G.node[u]['feat'] = np.array(G.node[u]['label'])

    random.seed(5)
    random.shuffle(graphs)
    N = len(graphs)
    N1 = int(N * 0.2)   #test idx
    test_graphs = [graphs[i] for i in range(0, N1)]
    graphs1 = [graphs[i] for i in range(N1, N)]
    del graphs
    graphs = graphs1
    train_idx = int(len(graphs) * args.train_ratio)
    train_graphs = graphs[:train_idx]
    val_graphs = graphs[train_idx:]
    #decide whether to select trees
    flag = 1
    if flag:
        s_time = time.time()
        train_trees = Select_Tree(train_graphs,k=args.num_trees,depth=args.depth,max_nodes=args.max_nodes)
        val_trees = Select_Tree(val_graphs, k=args.num_trees, depth=args.depth, max_nodes=args.max_nodes)
        test_tree = Select_Tree(test_graphs, k=args.num_trees, depth=args.depth, max_nodes=args.max_nodes)
        e_time =time.time()
        print("select time :{}".format(e_time-s_time))

        with open("sparse_train.txt",'w+') as f:
            f.write(str(train_trees))
        with open("sparse_test.txt",'w+') as f:
            f.write(str(val_trees))
    with open("sparse_train.txt",'r') as f:
        train_trees = eval(f.read())
    with open("sparse_test.txt",'r') as f:
        val_trees = eval(f.read())
    dataset_sampler = GraphSampler(train_graphs,train_trees,normalize=True, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs,val_trees, normalize=True, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, val_trees, normalize=True, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    feat_dim = dataset_sampler.feat_dim

    model = gcn_tree(feat_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        args.depth,concat = args.concat,bn=args.bn,dropout = args.dropout,alpha=args.alpha,args=args).cuda()

    print("start to train!")
    train(train_dataset_loader, model, args, val_dataset=val_dataset_loader,test_dataset=test_dataset_loader,\
          out_file=args.out_file,writer=writer)




