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
        loss = model.loss(ypred, label_)
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

def train(train_dataset, model, args, val_dataset, out_file="val.txt",test_dataset=None,writer=None,fold_id='0',print_flag=True):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=args.w_l2,amsgrad=True)
    iter = 0
    train_accs = []
    train_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        avg_l2 = 0.0
        model.train()
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

            l2 = 0
            for parm in model.parameters():
                l2 += torch.norm(parm, 2)
            avg_l2 += float(l2)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        avg_l2 /= (batch_idx+1)
        elapsed = time.time() - begin_time
        # if epoch<10:
        #    print("epoch: time",elapsed)
        result = evaluate(train_dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])

        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='test')
        if print_flag and epoch%4==0:
            print("epoch:{:3d},..time:{:.2f},...avg_loss:{:.3f},val_loss:{:.3f},...,train acc:{:.3f},...val acc:{:.3f}".format(epoch,
                                                                                                          elapsed,
                                                                                                          avg_loss,
                                                                                                          val_result['loss'],
                                                                                                          result['acc'],
                                                                                                          val_result['acc']))


        if writer != None:  #记录训练日志
            writer.add_scalars(fold_id+'/scalar/acc', {'train': result['acc'], 'val': val_result['acc']}, epoch)
            loss_all = avg_loss + avg_l2*args.w_l2
            writer.add_scalars(fold_id+'/scalar/ce_loss', {'train_loss':avg_loss,'val_loss':val_result['loss']}, epoch)
            writer.add_scalars(fold_id+'/scalar/train+ce+L2_loss',{'model_loss': loss_all, 'L2 loss': avg_l2*args.w_l2,\
                                                    'ce_loss':avg_loss }, epoch)

    return train_accs,val_accs



def cross_val(args,train_graphs,val_graphs,writer=None,fold_id='0'):
    print("start compute tree!!!")
    s_time = time.time()
    train_trees = Select_Tree(train_graphs, k=args.num_trees, depth=args.depth, max_nodes=args.max_nodes)
    val_trees = Select_Tree(val_graphs, k=args.num_trees, depth=args.depth, max_nodes=args.max_nodes)
    e_time = time.time()
    print("compute tree end! time cost:",e_time-s_time)

    print("prepare data for model !!!")
    dataset_sampler = GraphSampler(train_graphs, train_trees, normalize=True, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    del train_graphs
    del train_trees
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(val_graphs, val_trees, normalize=True, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    del val_graphs
    del val_trees
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    print("prepare train time cost:", time.time() - e_time,'load model')

    feat_dim = dataset_sampler.feat_dim

    model = gcn_tree(feat_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                     args.depth, pred_hidden_dims=[128, 32], concat=args.concat, bn=args.bn, dropout=args.dropout,
                     alpha=args.alpha, args=args).cuda()

    print("start to train!")
    train_acc, test_acc = train(train_dataset_loader, model, args, val_dataset=val_dataset_loader, test_dataset=None, \
          out_file=args.out_file, writer=writer,fold_id=fold_id)

    return train_acc, test_acc





if __name__=="__main__":
    args = arg_parse()
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    for G in graphs:
        for u in G.nodes():
            G.node[u]['feat'] = np.array(G.node[u]['label'])
    print("read file over!!")
    N = len(graphs)
    N1 = int(N *args.train_ratio)
    N2 = N-N1
    random.seed(args.seed)
    random.shuffle(graphs)
    train_accs = []
    val_accs = []
    #for k in range(10):
    k = int(args.fold_id)
    if k==-1:  #10 fold in a process run!! 
        k=0
    if k<10:
        val_start = k*N2
        k = k+1
        val_end = min(val_start + N2,N)
        val_graphs = [graphs[i] for i in range(val_start,val_end)]  #val set
        train_graphs = [graphs[i] for i in range(val_start)]
        for i in range(val_end,N):
            train_graphs.append(graphs[i])       #test set

        writer_path = args.file_name + "flod" + '-' + args.method + "-lr" + str(args.lr) + '-w' + str(
            args.w_l2) + '-drop' + str(args.dropout) + '-attention-' + str(args.attention)+"-batch_size-"\
            +str(args.batch_size) + 'tree_trans'+str(args.tree_trans)  # log dir

        print("log dir: "+writer_path)

        #try:  # 日志文件
        #    os.removedirs(writer_path)
        #except:
        #    pass
        writer = SummaryWriter(log_dir=writer_path)
        train_acc, val_acc = cross_val(args,train_graphs,val_graphs,writer=writer,fold_id=str(k))
        writer.close()
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        for n in range(len(train_acc)):
            print("fold:{:2d},epoch:{:3d}, train_acc:{:.3f}, val_acc:{:.3f}".format(k,n,train_acc[n],val_acc[n]))
        print("*************************************\n")
        if args.fold_id != -1:
            k=10
    train_accs_numpy = np.array(train_accs)
    val_accs_numpy = np.array(val_accs)
    try:
        np.save("result/"+str(args.fold_id)+args.method + "-train_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            train_accs_numpy)
        np.save("result/"+str(args.fold_id)+args.method + "-val_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            val_accs_numpy)
    except:
        np.save(str(args.fold_id)+args.method + "-train_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            train_accs_numpy)
        np.save(str(args.fold_id)+args.method + "-val_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            val_accs_numpy)
        
