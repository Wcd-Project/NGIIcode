from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
import pickle
from datetime import datetime

import utils.DataProcessing as DP
import utils.CalcHammingRanking as CalcHR

# from MODELS.resnet_cbam import *
# from Modeltest.resnet_cbam import * 
# from Modeltest.resnet_cbam_pc import * 

import CNN_model
# import CNN_model0
import numpy as np
import torch.nn as nn

# from MyAttention import *
# from MyCNN_model3 import *

# from center_loss import FocalLoss#,CenterLoss,
# from cnn_finetune import make_model

# from MODELS.dpn import *

# criterion_xent = nn.CrossEntropyLoss()


def LoadLabel(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def CreateModel(model_name, bit, use_gpu):
    if model_name == 'vgg11':
        vgg11 = models.vgg16(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bit)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bit)
    if model_name == 'resnet34':
        resnet = models.resnet34(pretrained=True)
        cnn_model = CNN_model.cnn_model(resnet, model_name, bit)
    if model_name == 'resnet50':
        resnet = models.resnet50(pretrained=True)
        cnn_model = CNN_model.cnn_model(resnet, model_name, bit)        
    if model_name == 'resnet34resnet34':
        resnet = models.resnet34(pretrained=True)
        cnn_model = CNN_model.cnn_model(resnet, model_name, bit)
		
    return cnn_model
    # return cnn_model
	
def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else: data_input = Variable(data_input)
        output= model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B

def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt
def Totloss(U, B, Sim, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta*theta).sum() / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(Variable(theta), False).data).sum()
    l2 = (U - B).pow(2).sum()
    l = l1 + lamda * l2
    return l, l1, l2, t1
def precision(trn_binary, trn_label, tst_binary, tst_label):
    #trn_binary = trn_binary.numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.numpy()
    #tst_binary = tst_binary.numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.numpy()
    query_times = tst_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)

    sum_p = np.zeros(trainset_len)
    sum_r = np.zeros(trainset_len)
    
    for i in range(query_times):
        print('Query ', i+1)
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)
        P = np.cumsum(buffer_yes) / Ns       
        R = np.cumsum(buffer_yes)/(trainset_len)*10
        sum_p = sum_p+P
        sum_r = sum_r+R
    
    return sum_p/1000,sum_r/1000
def DPSH_algo(bit, param, gpu_ind=0):
    # parameters setting
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)

    DATA_DIR = '/home/gpuadmin/datasets/CIFAR-10'
    DATABASE_FILE = 'database_img.txt'
    TRAIN_FILE = 'train_img.txt'
    TEST_FILE = 'test_img.txt'

    DATABASE_LABEL = 'database_label.txt'
    TRAIN_LABEL = 'train_label.txt'
    TEST_LABEL = 'test_label.txt'
    batch_size = 64      
    # batch_size = 256
    # batch_size = 150
    epochs = 200
    learning_rate = 0.05
    weight_decay = 10 ** -5
    # model_name = 'APN'
    # model_name = 'alexnet'
    # model_name = 'resnet18'
    # model_name = 'vgg11'
    # model_name = 'resnet34'

    model_name = 'resnet34resnet34'	
    # model_name = 'resnet50'

    # model_name = 'dpn92'
    # model_name = 'resnet50'
    # model_name = 'resnet50_cbam'
    # model_name = 'resnet18_cbam'
    # model_name = 'resnet34_cbam'
	# model_name = 'resnet18_pc'
    # model_name = 'resnet34_pc'
    # model_name = 'resnet101_cbam'
    # model_name = 'resnet152_cbam'
    print(model_name)
    nclasses = 10
    use_gpu = torch.cuda.is_available()

    filename = param['filename']

    lamda = param['lambda']
    param['bit'] = bit
    param['epochs'] = epochs
    param['learning rate'] = learning_rate
    param['model'] = model_name

    ### data processing
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),       
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dset_database = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, DATABASE_FILE, DATABASE_LABEL, transformations)

    dset_train = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, TRAIN_FILE, TRAIN_LABEL, transformations)

    dset_test = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, TEST_FILE, TEST_LABEL, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                             )

    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )

    ### create model
    model = CreateModel(model_name, bit, use_gpu)
    model = torch.nn.DataParallel(model).cuda()
    # model = AlexNetPlusLatent(bit).cuda()   
    print(model)    
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)


    ### training phase
    # parameters setting
    B = torch.zeros(num_train, bit)
    U = torch.zeros(num_train, bit)
    train_labels = LoadLabel(TRAIN_LABEL, DATA_DIR)
    train_labels_onehot = EncodingOnehot(train_labels, nclasses)
    test_labels = LoadLabel(TEST_LABEL, DATA_DIR)
    test_labels_onehot = EncodingOnehot(test_labels, nclasses)

    train_loss = []
    map_record = []

    totloss_record = []
    totl1_record = []
    totl2_record = []
    t1_record = []
    softmaxloss = torch.nn.CrossEntropyLoss().cuda()
    Sim = CalcSim(train_labels_onehot, train_labels_onehot)
    for epoch in range(epochs):
        epoch_loss = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_input, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            if use_gpu:
                train_label_onehot = EncodingOnehot(train_label, nclasses)
                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
                S = CalcSim(train_label_onehot, train_labels_onehot)
            else:
                train_label_onehot = EncodingOnehot(train_label, nclasses)
                train_input, train_label = Variable(train_input), Variable(train_label)
                S = CalcSim(train_label_onehot, train_labels_onehot)

            model.zero_grad()
            train_outputs= model(train_input)
            
            for i, ind in enumerate(batch_ind):
                U[ind, :] = train_outputs.data[i]
                B[ind, :] = torch.sign(train_outputs.data[i])

            Bbatch = torch.sign(train_outputs)
            # Bbatch1 = torch.sign(ym)
            if use_gpu:
                theta_x = train_outputs.mm(Variable(U.cuda()).t()) / 2
                logloss = (Variable(S.cuda())*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(train_label))#+(Bbatch1-ym).pow(2).sum() / (num_train * len(train_label))
            else:
                theta_x = train_outputs.mm(Variable(U).t()) / 2
                logloss = (Variable(S)*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(train_label))
            # l2loss = softmaxloss(c,train_label)#+ softmaxloss(cm,train_label)
            # Qloss = torch.mean((torch.abs(train_outputs) - 1) ** 2)

            # loss1 = nn.BCELoss()
            # output1 = loss1(nn.Sigmoid(c), train_label)

            # loss2 = nn.L1Loss()
            # output2 = loss2(c.cuda(), train_label)

            # loss3 = nn.MSELoss()
            # output3 = loss3(c.cuda(), train_label)
            # criterion_xent = nn.CrossEntropyLoss()
            # # criterion_cent = CenterLoss(nclasses, feat_dim=bit, use_gpu=use_gpu)
            # FocalLoss_loss = FocalLoss(nclasses, use_gpu)
            # loss_xent = criterion_xent(c, train_label)
            # loss_cent = criterion_cent(train_outputs, train_label)

            Qloss = torch.mean((torch.abs(train_outputs) - 1) ** 2) +torch.mean((torch.abs(Bbatch) - 1) ** 2) #+torch.mean((torch.abs(ym) - 1) ** 2)+torch.mean((torch.abs(Bbatch1) - 1) ** 2)   #  +torch.mean((torch.abs(Bbatch - train_outputs) ) ** 2)         
            loss =  - logloss + lamda * regterm#+l2loss+Qloss   # +   FocalLoss_loss       #     #    +loss_xent+loss_cent#+loss2+loss3#+loss1#
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # print('[Training Phase][Epoch: %3d/%3d][Iteration: %3d/%3d] Loss: %3.5f' % \
            #       (epoch + 1, epochs, iter + 1, np.ceil(num_train / batch_size),loss.data[0]))
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader)), end='')
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

        l, l1, l2, t1 = Totloss(U, B, Sim, lamda, num_train)
        totloss_record.append(l)
        totl1_record.append(l1)
        totl2_record.append(l2)
        t1_record.append(t1)

        print('[Total Loss: %10.5f][total L1: %10.5f][total L2: %10.5f][norm theta: %3.5f]' % (l, l1, l2, t1), end='')

        ### testing during epoch
        qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
        #tB = torch.sign(B).numpy()
        tB = GenerateCode(model, train_loader, num_train, bit, use_gpu)
        map_ = CalcHR.CalcTopMap(qB, tB, test_labels_onehot.numpy(), train_labels_onehot.numpy(),5000)
        train_loss.append(epoch_loss / len(train_loader))
        map_record.append(map_)

        print('[Test Phase ][Epoch: %3d/%3d] MAP(retrieval train): %3.5f' % (epoch+1, epochs, map_))
        print(len(train_loader))
    ### evaluation phase
    ## create binary code
    model.eval()
    database_labels = LoadLabel(DATABASE_LABEL, DATA_DIR)
    database_labels_onehot = EncodingOnehot(database_labels, nclasses)
    qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
    dB = GenerateCode(model, database_loader, num_database, bit, use_gpu)

    map = CalcHR.CalcTopMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(),5000)
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
    # print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
    test_binary = 0.5*(qB+1)
    database_binary = 0.5*(dB+1)
    test_labels = LoadLabel(TEST_LABEL, DATA_DIR)
    database_labels = LoadLabel(DATABASE_LABEL, DATA_DIR)
    sum_p,sum_r = precision(database_binary, database_labels, test_binary, test_labels)
    if not os.path.isdir('result/'+str(bit)):
            os.mkdir('result/'+str(bit))
    np.savetxt ('./result'+'/'+str(bit)+'/'+'sum_p.txt',sum_p,fmt='%3.5f')  
    np.savetxt ('./result'+'/'+str(bit)+'/'+'sum_r.txt',sum_r,fmt='%3.5f')
    np.savetxt('./result'+'/'+str(bit)+'/'+'map', map.reshape(1,-1),fmt='%3f')
    result = {}
    result['qB'] = qB
    result['dB'] = dB
    result['train loss'] = train_loss
    result['map record'] = map_record
    result['map'] = map
    result['param'] = param
    result['total loss'] = totloss_record
    result['l1 loss'] = totl1_record
    result['l2 loss'] = totl2_record
    result['norm theta'] = t1_record
    result['filename'] = filename
    # if not os.path.isdir('result'):
       # os.mkdir('result')
    # torch.save(qB, './result/qB.txt')
    # torch.save(dB, './result/dB.txt')
    # torch.save(map, './result/map.txt')
  #  np.savetxt('./result/qB.txt', qB,fmt='%d',delimiter=',')
    
    return result

if __name__=='__main__':
    bit = 12
    lamda = 50
    gpu_ind = 0
    filename = 'log/DPSH_' + str(bit) + 'bits_NUS-WIDE_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
    param = {}
    param['lambda'] = lamda
    param['filename'] = filename
    result = DPSH_algo(bit, param, gpu_ind)
    fp = open(result['filename'], 'wb')
    pickle.dump(result, fp)
    fp.close()

