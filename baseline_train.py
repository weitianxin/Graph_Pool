import GraphCnn
import torch
import read_data
import torch.utils.data
import graph_sampler
from torch.autograd import Variable
import numpy as np
import sklearn.metrics as metrics
import random
import arg_parse
from tensorboardX import SummaryWriter
import os


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
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feature'].float()).to(device)
        num_nodes = Variable(data['num_nodes'].float(), requires_grad=False).to(device)
        label_ = Variable(data['label'].long(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        ypred = model(adj,h0,number_nodes=num_nodes)
        loss = model.loss(ypred,label_)
        avg_loss += loss
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
    # if name=='test':
    #     print(name, " pre:", preds)
    return result

def train(model,dataset,device,valdataset,testdataset=None,num_epochs=1000,lr=0.001,writer=None,w_l2=0.0001):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=w_l2)
    iter = 0
    r1 = evaluate(dataset, model, device, name='train')
    r2 = evaluate(valdataset, model, device, name='test')
    print("epoch:{},...train acc:{},...test acc:{}".format(-1,r1['acc'], r2['acc']))


    for epoch in range(num_epochs):
        avg_loss = 0
        avg_l2 = 0
        selfl2 = True
        model.train()
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long(), requires_grad=False).to(device)
            h0 = Variable(data['feature'].float(), requires_grad=False).to(device)
            num_nodes = Variable(data['num_nodes'].float(), requires_grad=False).to(device)

            y_pred = model(adj, h0,number_nodes=num_nodes)
            loss = model.loss(y_pred, label)

            if selfl2 == True:  # L2 norm by myself
                l2 = 0
                for parm in model.parameters():
                    l2 += torch.norm(parm, 2)
                avg_l2 += l2
                #loss = loss + l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            iter += 1
            avg_loss += loss
        avg_loss /= (batch_idx + 1)
        avg_l2 /= (batch_idx+1)
        #print('epoch:{},avg_loss:{}'.format(epoch,avg_loss))
        r1 = evaluate(dataset,model,device,name='train')
        r2 = evaluate(valdataset,model,device,name='val')
        if testdataset !=None:
            r3 = evaluate(testdataset,model,device,name='test')
            print("epoch:{}, ...avg_loss:{:.4f}, ...train acc:{:.3f}, ...val acc:{:.3f},\
             ...test acc:{:.3f}".format(epoch,avg_loss,r1['acc'],r2['acc'],r3['acc']))
        else:
            print("epoch:{}, ...avg_loss:{:.4f}, ...train acc:{:.3f}, ...val acc:{:.3f},".format(epoch, avg_loss, r1['acc'], r2['acc']))

        if writer != None:
            writer.add_scalars('scalar/acc', {'train': r1['acc'], 'val': r2['acc']}, epoch)
            loss_nol2 = avg_loss + avg_l2 * w_l2
            if testdataset!=None:
                writer.add_scalars('scalat/ce_loss', {'train_loss': r1['loss'], 'val_loss': r2['loss']}, epoch)
            else:
                writer.add_scalars('scalat/ce_loss', {'train_loss':r1['loss'],'val_loss':r2['loss']}, epoch)
            writer.add_scalars('scalat/train+ce+L2_loss',{'model_loss': loss_nol2, 'L2 loss': avg_l2 * w_l2,\
                                                    'ce_loss':avg_loss }, epoch)
def main():
    arg = arg_parse.arg_parse()
    writer_path = arg.method + "-lr" + str(arg.lr) + '-w' + str(arg.w_l2)
    print(writer_path)
    try:                             #日志文件
        os.removedirs(writer_path)
    except:
        pass
    writer = SummaryWriter(log_dir=writer_path)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    Gs = read_data.read_graphfile('data','DD')
    print("read from file over!")
    random.seed(1)
    random.shuffle(Gs)
    N = len(Gs)
    N1 = int(N*0.4)  #0.8
    #test_Gs = [Gs[i] for i in range(N1, N)]  # 用以test
    Gs = [Gs[i] for i in range(N1)]
    train_dataset,val_dataset = prepare_data(Gs, rate=0.75, batch_size=arg.batch_size, number_works=arg.num_workers)
    feature_dim = len(Gs[0].node[0]['feature'])
    del Gs
    model = GraphCnn.GCNs(feature_dim, num_layers=2, hidden_dim=64, class_dim=2, pooling_type=arg.method,drop_rate=0.5).to(device)#64
    print("start training!!")
    # data_sampler = graph_sampler.GraphSampler(test_Gs)
    # test_dataloader = torch.utils.data.DataLoader(data_sampler,
    #                                                batch_size=arg.batch_size,
    #                                                shuffle=False,
    #                                                num_workers=arg.num_workers)

    train(model, train_dataset, device, val_dataset,testdataset=None, num_epochs=arg.num_epochs, lr=arg.lr, writer=writer, w_l2=arg.w_l2)#0.001
    writer.close()

if __name__=="__main__":
    main()

