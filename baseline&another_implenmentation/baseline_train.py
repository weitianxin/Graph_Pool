import GraphCnn
import torch
import read_data
import load_data
import torch.utils.data
import graph_sampler
from torch.autograd import Variable
import numpy as np
import sklearn.metrics as metrics
import random
import arg_parse
from tensorboardX import SummaryWriter
import os
import time
import THAN


def prepare_data(Gs,rate=0.8,batch_size=20,number_works=0):
    random.seed(22)
    random.shuffle(Gs)
    N = len(Gs)
    test_start_id =int(N*rate)
    train_graphs = [Gs[i] for i in range(test_start_id)]
    test_graphs = [Gs[i] for i in range(test_start_id,N)]

    data_sampler = graph_sampler.GraphSampler(train_graphs)
    train_dataloader = torch.utils.data.DataLoader(data_sampler,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=number_works)
    test_data_sampler = graph_sampler.GraphSampler(test_graphs)
    test_dataloader = torch.utils.data.DataLoader(test_data_sampler,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=number_works)
    return train_dataloader,test_dataloader

def evaluate(dataset, model, device,name='Validation'): #evalute
    model.eval()
    labels = []
    preds = []
    avg_loss = 0
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feature'].float()).cuda()
        num_nodes = Variable(data['num_nodes'].float(), requires_grad=False).cuda()
        #print(num_nodes.shape)
        label_ = Variable(data['label'].long(), requires_grad=False).cuda()
        labels.append(data['label'].long().numpy())
        ypred = model(adj,h0,number_nodes=num_nodes)
        loss = model.loss(ypred,label_)
        avg_loss += float(loss)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    avg_loss /= (batch_idx+1)
    result = {'loss':avg_loss,
             # 'prec': metrics.precision_score(labels, preds, average='macro'),
             # 'recall': metrics.recall_score(labels, preds, average='macro'),
             # 'F1': metrics.f1_score(labels, preds, average="micro")
              'acc': metrics.accuracy_score(labels, preds)
             }

    return result

def train(arg,model,dataset,valdataset,device,testdataset=None,writer=None):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=arg.lr,weight_decay=arg.w_l2)
    iter = 0
    r1 = evaluate(dataset, model, device)
    r2 = evaluate(valdataset, model, device)
    print("epoch:{},...train acc:{},...val acc:{}".format(-1,r1['acc'], r2['acc']))

    train_acc = []
    val_acc = []
    s_time = time.time()
    selfl2 = True
    for epoch in range(arg.num_epochs):
        avg_loss = 0
        avg_l2 = 0
        model.train()
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long(), requires_grad=False).to(device)
            h0 = Variable(data['feature'].float(), requires_grad=False).to(device)
            num_nodes = Variable(data['num_nodes'].float(), requires_grad=False).to(device)
            #num_nodes = data['num_nodes'].long().numpy()
            #print(num_nodes.shape)
            y_pred = model(adj, h0,number_nodes=num_nodes)
            loss = model.loss(y_pred, label)

            if selfl2 == True:  # L2 norm by myself
                l2 = 0
                for parm in model.parameters():
                    l2 += torch.norm(parm, 2)
                avg_l2 += float(l2)
                #loss = loss + l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            iter += 1
            avg_loss += float(loss)
        avg_loss /= (batch_idx + 1)
        avg_l2 /= (batch_idx+1)
        
        r1 = evaluate(dataset,model,device,name='train')
        r2 = evaluate(valdataset,model,device,name='val')
        train_acc.append(r1['acc'])
        val_acc.append(r2['acc'])
        if testdataset !=None:
            r3 = evaluate(testdataset,model,device,name='test')
        else:
            pass
        print("epoch:{:3d}, ...avg_loss:{:.4f},...val_loss:{:.4f} ...train acc:{:.3f}, ...val acc:{:.3f},".format(epoch, avg_loss, r2['loss'], r1['acc'], r2['acc']))
        if writer != None:
            writer.add_scalars('scalar/acc', {'train': r1['acc'], 'val': r2['acc']}, epoch)
            loss_nol2 = avg_loss + avg_l2 * arg.w_l2
            writer.add_scalars('scalat/ce_loss', {'train_loss':r1['loss'],'val_loss':r2['loss']}, epoch)
            writer.add_scalars('scalat/train+ce+L2_loss',{'model_loss': loss_nol2, 'L2 loss': avg_l2 * arg.w_l2,\
                                                    'ce_loss':avg_loss }, epoch)
        if epoch==9:
            print("10 epoch time cost:",time.time()-s_time)
    return train_acc,val_acc


def cross_val(arg,feature_dim,train_graphs,val_graphs,device,writer=None):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    s_time = time.time()
    print("prepare data!!!")
    if arg.method != 'others':
       print("not tree method")
    else:
        print("*****************************************************************")
        print("please used train.py ! we will change it to use average model!")
        print("*****************************************************************")
        arg.method = 'average'

    data_sampler = graph_sampler.GraphSampler(train_graphs)
    del train_graphs
    train_dataloader = torch.utils.data.DataLoader(data_sampler,
                                                   batch_size=arg.batch_size,
                                                   shuffle=False,
                                                   num_workers=arg.num_workers)
    val_data_sampler = graph_sampler.GraphSampler(val_graphs)
    del val_graphs
    val_dataloader = torch.utils.data.DataLoader(val_data_sampler,
                                                  batch_size=arg.batch_size,
                                                  shuffle=False,
                                                  num_workers=arg.num_workers)
    print("prepare deata cost time:",time.time()-s_time)

    print("creat Model!!!")
    model = GraphCnn.GCNs(feature_dim, num_layers=arg.num_gcn_layers, hidden_dim=arg.hidden_dim, class_dim=2, pooling_type=arg.method,
                          drop_rate=arg.dropout,average_flag=arg.avg_type).to(device)  # 64
    print("start train!!!")

    train_acc,val_acc = train(arg,model,train_dataloader,val_dataloader,device,writer=writer)
    return train_acc,val_acc

if __name__=="__main__":
    args = arg_parse.arg_parse()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    graphs = load_data.read_graphfile('data', 'DD', max_nodes=args.max_nodes)
    #graphs = graphs[0:50]
    for G in graphs:
        for u in G.nodes():
            G.node[u]['feat'] = np.array(G.node[u]['label'])
    feature_dim = len(graphs[0].node[0]['feat'])
    print("read file over!!")
    N = len(graphs)
    N1 = int(N *args.train_ratio)
    N2 = N-N1
    random.seed(args.seed)
    random.shuffle(graphs)
    train_accs = []
    val_accs = []
    k = int(args.fold_id)
    if k == -1:
        k = 0
    if k < 10:
        val_start = k*N2
        val_end = min(N,val_start + N2)
        val_graphs = [graphs[i] for i in range(val_start,val_end)]  #val set
        train_graphs = [graphs[i] for i in range(val_start)]
        for i in range(val_end,N):
            train_graphs.append(graphs[i])       #test set

        writer_path =args.file_name + "flod" + str(k) + '-' + args.method + "-lr" + str(args.lr) + '-w' + str(
            args.w_l2) + '-drop' + str(args.dropout) + '-attention' + str(args.attention)+'-tre_num'+str(args.tree_num)  # log dir

        print("log dir: "+writer_path)

        try:  # 日志文件
            os.removedirs(writer_path)
        except:
            pass
        writer = SummaryWriter(log_dir=writer_path)
        train_acc, val_acc = cross_val(args, feature_dim, train_graphs, val_graphs, device, writer = writer)
        writer.close()
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if args.fold_id != -1 :
            k = 10
        else:
            k += 1
    train_accs_numpy = np.array(train_accs)
    val_accs_numpy = np.array(val_accs)

    np.save(args.method + "-train_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            train_accs_numpy)
    np.save(args.method + "-val_acc-" + str(args.lr) + "-" + str(args.w_l2) + '-' + str(args.dropout),
            val_accs_numpy)

