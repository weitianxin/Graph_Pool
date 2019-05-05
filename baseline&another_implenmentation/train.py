import THAN
import creat_trees
import numpy as np
import torch
import read_data
import load_data
import torch.utils.data
from torch.autograd import Variable
import sklearn.metrics as metrics
import torch.nn as nn
import random
import arg_parse
from tensorboardX import SummaryWriter
import os
import time
import torch.backends.cudnn
import graph_sampler
import torch.nn.functional as F


def pre_val_data(arg,graphs,train_rate=0.8,max_depth=4): # creat train and test sets
    batch_size = arg.batch_size
    num_workers = arg.num_workers
    tree_num = arg.tree_num
    N=len(graphs)
    random.seed(22)
    random.shuffle(graphs)
    start_test = int(N*train_rate)
    train_set = [graphs[i] for i in range(start_test)]
    test_set = [graphs[i] for i in range(start_test,N)]
    data_sampler = creat_trees.graph_sampler(train_set, max_depth=max_depth, max_node=arg.max_nodes, tree_num=tree_num)
    train_set_loader = torch.utils.data.DataLoader(data_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

    data_sampler = creat_trees.graph_sampler(test_set, max_depth=max_depth, max_node=arg.max_nodes, tree_num=tree_num)
    test_set_loader = torch.utils.data.DataLoader(data_sampler,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    return train_set_loader, test_set_loader


def evaluate(dataset, model, device,name='Val', max_num_examples=None): #evalute
    model.eval()
    labels = []
    preds = []
    avg_loss = 0
    for batch_idx, data in enumerate(dataset):

        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feature'].float()).to(device)
        label_ = Variable(data['label'].long()).to(device)
        labels.append(data['label'].long().numpy())
        tree_info = data['tree_info'].float().to(device)


        ypred = model(adj, h0, tree_info)
        if name=='val':
            loss = F.cross_entropy(ypred, label_)
            avg_loss += float(loss)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    avg_loss /=(batch_idx+1)

    result = {'loss': avg_loss,
              'acc': metrics.accuracy_score(labels, preds)
              }
    return result

def train(arg, model, train_set, test_set, device,testdataset=None, writer=None,print_flag=True,fold_id='0'):
    selfl2 = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),\
                                 lr=arg.lr, weight_decay=arg.w_l2)
    train_acc = []
    val_acc = []
    val_loss = []
    for epoch in range(arg.num_epochs):
        begin_time = time.time()
        avg_loss = 0
        avg_l2 = 0
        model.train()
        all_time = 0
        for batch_idx, data in enumerate(train_set):
            model.zero_grad()
            l_time = time.time()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long(), requires_grad=False).cuda()
            h0 = Variable(data['feature'].float(), requires_grad=False).cuda()
            tree_info = Variable(data['tree_info'].float(), requires_grad=False).cuda()
            #print("load data time cost:",time.time()-begin_time)

            y_pred = model(adj, h0, tree_info)
            loss = model.loss(y_pred, label)
            all_time += time.time()-l_time

            if selfl2 == True:                    #L2 norm by myself
                l2 = 0
                for parm in model.parameters():
                    l2 += torch.norm(parm, 2)
                avg_l2 += float(l2)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), arg.clip)
            optimizer.step()
            avg_loss += float(loss)

        #print("load time",all_time)
        avg_loss /= (batch_idx+1)
        avg_l2 /= (batch_idx+1)
        #print("train time:",time.time()-begin_time)
        r1 = evaluate(train_set, model, device, name='train')
        r2 = evaluate(test_set, model, device, name='val')
        train_acc.append(r1['acc'])
        val_acc.append(r2['acc'])
        val_loss.append(r2['loss'])
        #print("acc time:",time.time()-begin_time)

        # if testdataset is not None:
        #     r3 = evaluate(test_set,model,device,name='test')

        if writer != None:
            writer.add_scalars(fold_id+'/scalar/acc', {'train': r1['acc'], 'val': r2['acc']}, epoch)
            loss_nol2 = avg_loss + avg_l2 * arg.w_l2
            writer.add_scalars(fold_id+'scalar/ce_loss', {'train_loss': avg_loss, 'val_loss': r2['loss']}, epoch)
            writer.add_scalars(fold_id+'scalar/train+ce+L2_loss', {'model_loss': loss_nol2, 'L2 loss': avg_l2 * arg.w_l2, \
                                                           'ce_loss': avg_loss}, epoch)

        if print_flag:
            elapsed = time.time() - begin_time
            print("epoch:{},...time:{:3.2f}, ...avg_loss:{:.4f}, val_loss:{:.4f},...train acc:{:.3f}, ...val acc:{:.3f}".format(epoch,
                                                                                                                elapsed,
                                                                                                                avg_loss,
                                                                                                                r2['loss'],
                                                                                                                r1['acc'],
                                                                                                                r2['acc']))
    return train_acc,val_acc,val_loss

def cross_val(arg,feature_dim,train_graphs,val_graphs,device,writer=None,fold_id=0):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    max_depth = arg.max_depth
    s_time = time.time()
    print("prepare data!!!")
    if arg.method == 'others':
       print("tree method")
    else:
        print("*****************************************************************************")
        print("please used baseline_train.py,and we will use the tree model insead of "+arg.method+"!!")
        print("*****************************************************************************")
        arg.method = 'others'


    data_sampler = creat_trees.graph_sampler(train_graphs, max_depth=max_depth, max_node=arg.max_nodes,
                                             tree_num=arg.tree_num)
    train_set_loader = torch.utils.data.DataLoader(data_sampler,
                                                  batch_size=arg.batch_size,
                                                  shuffle=False,
                                                  num_workers=arg.num_workers)
    data_sampler = creat_trees.graph_sampler(val_graphs, max_depth=max_depth, max_node=arg.max_nodes,
                                             tree_num=arg.tree_num)
    val_set_loader = torch.utils.data.DataLoader(data_sampler,
                                                   batch_size=arg.batch_size,
                                                   shuffle=False,
                                                   num_workers=arg.num_workers)
    print("prepare deata cost time:",time.time()-s_time)

    #model = THAN.THAN_ADJ(input_feature_dim=feature_dim,hidden_dim=hidden_dim,max_depth=max_depth)  #not gcn

    print("creat Model!!!")
    model = THAN.TreeGCN(feature_dim,attention=arg.attention, num_layers=arg.num_gcn_layers, hidden_dim=arg.hidden_dim, class_dim=arg.num_classes, pre_hidden_dim=[128,32], pooling_type=arg.method,\
                            drop_rate=arg.dropout, max_depth=max_depth, need_feature_chang=False).cuda()
    print("start train!!!")

    train_acc,val_acc,val_loss = train(arg,model,train_set_loader,val_set_loader,device, testdataset=None, writer=writer,fold_id=str(fold_id))

    return train_acc,val_acc,val_loss

if __name__=="__main__":
    args = arg_parse.arg_parse()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    graphs = load_data.read_graphfile('data', 'DD', max_nodes=args.max_nodes)
    # random.shuffle(graphs)
    # graphs = graphs[0:50]  #*******************************
    max_N = 0
    min_N = 1000
    for G in graphs:
        Ni = G.number_of_nodes()
        max_N = max(max_N,Ni)
        min_N = min(min_N,Ni)
        for u in G.nodes():
            G.node[u]['feat'] = np.array(G.node[u]['label'])
    print("min nodes num:{},max nodes num:{}".format(min_N,max_N))
    feature_dim = len(graphs[0].node[0]['feat'])
    print("read file over!!")
    N = len(graphs)
    N1 = int(N * args.train_ratio)
    N2 = N-N1
    random.seed(args.seed)
    random.shuffle(graphs)
    train_accs = []
    val_accs = []
    k = args.fold_id
    if k == -1:    #10 fold run togehter
        k = 0

    while(k<10):
        val_start = k*N2
        val_end = min(N,val_start + N2)
        val_graphs = [graphs[i] for i in range(val_start,val_end)]  #val set
        train_graphs = [graphs[i] for i in range(val_start)]
        for i in range(val_end,N):
            train_graphs.append(graphs[i])       #test set

        writer_path = args.file_name + "flod" + '-' + args.method + "-lr" + str(args.lr) + '-w' + str(
            args.w_l2) + '-drop' + str(args.dropout) + '-attention' + str(args.attention)+'-tre_num'+str(args.tree_num)  # log dir

        print("log dir: "+writer_path)
        #
        # try:  # 日志文件
        #     os.removedirs(writer_path)
        # except:
        #     pass
        # writer = SummaryWriter(log_dir=writer_path)

        writer = SummaryWriter(log_dir=writer_path)
        train_acc, val_acc, val_loss = cross_val(args, feature_dim, train_graphs, val_graphs, device, writer = writer, fold_id=k)
        writer.close()
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if args.fold_id != -1:   # only run a fold
            k = 10
        else:
            k += 1
    train_accs_numpy = np.array(train_accs)
    val_accs_numpy = np.array(val_accs)
    val_loss_numpy = np.array(val_loss)

    np.save("/userhome/THAN_CR/result/"+args.method + "-train_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            train_accs_numpy)
    np.save("/userhome/THAN_CR/result/"+args.method + "-val_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            val_accs_numpy)
    np.save("/userhome/THAN_CR/result/" + args.method + "-val_loss-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(
        args.dropout),val_loss_numpy)
