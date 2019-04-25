import THAN
import creat_trees
import numpy as np
import torch
import read_data
import random
import torch.utils.data
from torch.autograd import Variable
import sklearn.metrics as metrics
import torch.nn as nn
import random
import arg_parse
from tensorboardX import SummaryWriter
import os
import time


def pre_val_data(graphs,train_rate=0.8,max_depth=4): # creat train and test sets
    batch_size = 20
    num_workers = 1
    N=len(graphs)
    random.seed(0)
    random.shuffle(graphs)
    start_test = int(N*train_rate)
    train_set = [graphs[i] for i in range(start_test)]
    test_set = [graphs[i] for i in range(start_test,N)]
    data_sampler = creat_trees.graph_sampler(train_set,max_depth=max_depth)
    train_set_loader = torch.utils.data.DataLoader(data_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

    data_sampler = creat_trees.graph_sampler(test_set, max_depth=max_depth)
    test_set_loader = torch.utils.data.DataLoader(data_sampler,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    return train_set_loader, test_set_loader


def evaluate(dataset, model, device,name='Validation', max_num_examples=None): #evalute
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
        loss = model.loss(ypred, label_)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        avg_loss += float(loss)

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    avg_loss /=(batch_idx+1)

    result = {'loss':avg_loss,
              #'prec': metrics.precision_score(labels, preds, average='macro'),
              #'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds)
              #'F1': metrics.f1_score(labels, preds, average="micro"),
              }
    #print(name, " accuracy:", result['acc'])
    return result

def train(model,train_set,test_set,testdataset=None,num_epochs=50,lr=0.001,writer=None,w_l2=0.0001):
    selfl2 = False
    if torch.cuda.is_available():
        device=torch.device("cuda:0")
    else:
        device=torch.device("cpu")
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),\
                                 lr=lr,weight_decay=w_l2)
    iter = 0
    for epoch in range(num_epochs):
        begin_time = time.time()
        avg_loss = 0
        avg_l2 = 0
        model.train()
        for batch_idx, data in enumerate(train_set):
            model.zero_grad()
            adj = Variable(data['adj'].float(),requires_grad=False).to(device)
            label = Variable(data['label'].long(),requires_grad=False).to(device)
            h0 = Variable(data['feature'].float(),requires_grad=False).to(device)
            tree_info = Variable(data['tree_info'].float(),requires_grad=False).to(device)
            y_pred = model(adj,h0,tree_info)
            # y_pred = torch.squeeze(y_pred,-2)
            # L2 loss
            loss = model.loss(y_pred, label)
            if selfl2==True:  #L2 norm by myself
                l2 = 0
                for parm in model.parameters():
                    l2 += torch.norm(parm,2)
                avg_l2 += l2
                loss = loss + l2

            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += float(loss)
        # del adj
        # del tree_info
        # del y_pred
        avg_loss /= (batch_idx+1)
        avg_l2 /= (batch_idx+1)
        elapsed = time.time() - begin_time
        #print("epoch:{}, ...Loss:{}".format(epoch,avg_loss))
        r1 = evaluate(train_set,model,device,name='train')
        r2 = evaluate(test_set,model,device,name='val')
        if testdataset is not None:
            r3 = evaluate(test_set,model,device,name='test')
            print("epoch:{},...time:{:3.2f}, ...avg_loss:{:.4f}, ...train acc:{:.3f},\
             ...val acc:{:.3f},...test acc:{:.2f}".format(epoch, elapsed,avg_loss, r1[  'acc'], r2[ 'acc'],r3['acc']))
        else:
            print("epoch:{},...time:{:3.2f}, ...avg_loss:{:.4f}, ...train acc:{:.3f}, ...val acc:{:.3f}".format(epoch,elapsed, avg_loss, r1['acc'],
                                                                                             r2['acc']))

        if writer != None:
            writer.add_scalars('scalar/acc', {'train': r1['acc'], 'val': r2['acc']}, epoch)
            loss_nol2 = avg_loss + avg_l2 * w_l2
            if testdataset != None:
                writer.add_scalars('scalat/ce_loss', {'train_loss': r1['loss'], 'val_loss': r2['loss']}, epoch)
            else:
                writer.add_scalars('scalat/ce_loss', {'train_loss': r1['loss'], 'val_loss': r2['loss']}, epoch)
            writer.add_scalars('scalat/train+ce+L2_loss', {'model_loss': loss_nol2, 'L2 loss': avg_l2 * w_l2, \
                                                           'ce_loss': avg_loss}, epoch)

def main():
    arg = arg_parse.arg_parse()
    writer_path = arg.method + "-lr" + str(arg.lr) + '-w' + str(arg.w_l2)
    print(writer_path)
    try:  # 日志文件
        os.removedirs(writer_path)
    except:
        pass
    writer = SummaryWriter(log_dir=writer_path)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    graphs_ = read_data.read_graphfile('data','DD')
    random.shuffle(graphs_ )
    N = len(graphs_)
    N1 = int(N*0.8)
    test_graphs =[graphs_[i] for i in range(N1,N)]
    graphs = [graphs_[i] for i in range(N1)]  #暂时只选取100个
    del graphs_
    feature_dim = len(graphs[0].node[0]['feature'])
    hidden_dim = 64
    max_depth = 4
    train_set,val_set = pre_val_data(graphs,train_rate=0.75,max_depth=max_depth)
    #model = THAN.THAN_ADJ(input_feature_dim=feature_dim,hidden_dim=hidden_dim,max_depth=max_depth)  #not gcn
    print("creat Model!!!")
    model = THAN.THANaddGCN(feature_dim,num_layers=2,hidden_dim=64,class_dim=2,pooling_type=arg.method,\
                            drop_rate=arg.dropout,max_depth=4,need_feature_chang=False).to(device)
    print("start train!!!")

    data_sampler = creat_trees.graph_sampler(train_set, max_depth=max_depth)
    test_set_loader = torch.utils.data.DataLoader(data_sampler,
                                                   batch_size=20,
                                                   shuffle=False,
                                                   num_workers=1)

    train(model,train_set,val_set,testdataset=test_set_loader,num_epochs=100,lr=0.001,writer=writer)

if __name__ == '__main__':
    print('start !!!*****************************************')
    main()

