from utils import *
from utils.tools import *
from models.dpsh import *
from utils.evaluate import *
import sys
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import time
import os
from loguru import logger
import numpy as np
plt.switch_backend('agg')
torch.multiprocessing.set_sharing_strategy('file_system')
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
# from metric import  calculate_top_map, calculate_map, p_topK, pr_curve, calc_map_k
from utils.evaluate1 import *
# f=open('log.txt', 'w+')
import random
def get_config():
    config = {
        "alpha": 0.1,
        "lambda": 0.0001,
        "alpha1": 0.5,
        "gamma1": 10,
        "lambda1": 0.1,

        #"optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}, "lr_type": "step"},
        #"optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        # "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "F029GreedyHash-DHN-IDHN-CSQ-DTSHrs",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        # "batch_size": 32,


        "net": AlexNet,
        # "net":ResNet,


        # "dataset": "KIMIAPath960",
        # "dataset": "KIMIAPath24",
	
        # "dataset": "nuswide",
        #"dataset": "nuswide_21",
        #"dataset":"coco",
        #"dataset":"nuswide_81",
        # "dataset":"imagenet",


        # "dataset": "AID",
        # "dataset": "UC_Merced",
        # "dataset": "UC_Merced",
        # "dataset": "cifar10",
        # "dataset": "KIMIAPath24",
        # "dataset": "cifar10",
        # "dataset": "FDXJ",
        "dataset": "UC_Merced",

        "epoch": 200,
        "test_map": 200,
        "GPU": True,
        # "GPU":False,
        # "bit_list": [16,24,32,36,48,64,96,128,256],
        # "bit_list": [8,16,32,64,128,256],  #2^n
        "bit_list": [32],
    }


    # if config["dataset"] == "AID":
        # config["topK"] = 60
        # config["n_class"] = 30
        # # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/" + config["dataset"] + "/"
        # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/" + config["dataset"] + "/"

    # # if config["dataset"] == "UC_Merced":
        # # config["topK"] = 2100
        # # config["n_class"] = 21
        # # # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/" + config["dataset"] + "/"
        # # config["data_path"] = "/home/gpuadmin/RS20200817Demo/RsDatasets/" + config["dataset"] + "/"


    # if config["dataset"] == "KIMIAPath960":
        # config["topK"] = 50
        # config["n_class"] = 20
        # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/K960/"


    # config["data"] = {
        # "train_set": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K960MAP50/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        # "database": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K960MAP50/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        # "test": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K960MAP50/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}


    # if config["dataset"] == "KIMIAPath24":
        # config["topK"] = 50
        # config["n_class"] = 24
        # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/K24/"


    # config["data"] = {
        # "train_set": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K24MAP50/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        # "database": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K24MAP50/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        # "test": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K24MAP50/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}


    # if config["dataset"] == "nuswide":
        # config["topK"] = 5000
        # config["n_class"] = 21
        # config["data_path"] = "/home/csl1/data3/nuswide/"


    # config["data"] = {
        # "train_set": {"list_path": "/home/csl1/data3/Nuswide21/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        # "database": {"list_path": "/home/csl1/data3/Nuswide21/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        # "test": {"list_path": "/home/csl1/data3/Nuswide21/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}

    # if config["dataset"] == "KIMIAPath24":
        # config["topK"] = 50
        # config["n_class"] = 24
        # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/K24/"


    # config["data"] = {
        # "train_set": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K24MAP50/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        # "database": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K24MAP50/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        # "test": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/K24MAP50/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}

    # if config["dataset"] == "cifar10":
        # config["topK"] = 12
        # config["n_class"] = 10
        # config["data_path"] = "/home/csl/RS20200817Demo/RsDatasets/" + config["dataset"] + "/"


    # config["data"] = {
        # "train_set": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/CifarDaLunWen/MAP1000/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        # "database": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/CifarDaLunWen/MAP1000/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        # "test": {"list_path": "/home/csl/RS20200817Demo/RsDatasets/CifarDaLunWen/MAP1000/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}

    if config["dataset"] == "UC_Merced":
        config["topK"] = 50
        config["n_class"] = 21
        config["data_path"] = "/home/gpuadmin/ZNDMT8-20201107/ZNSPSSFXXT2/UC_Merced/"


    config["data"] = {
        "train_set": {"list_path": "/home/gpuadmin/ZNDMT8-20201107/ZNSPSSFXXT2/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "/home/gpuadmin/ZNDMT8-20201107/ZNSPSSFXXT2/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "/home/gpuadmin/ZNDMT8-20201107/ZNSPSSFXXT2/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


class GreedyHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(GreedyHashLoss, self).__init__()
        self.fc = torch.nn.Linear(bit, config["n_class"], bias=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        if config["GPU"]:
            self.fc = self.fc.cuda()
            self.criterion = self.criterion.cuda()

    def forward(self, u, onehot_y, ind, config):
        b = GreedyHashLoss.Hash.apply(u)
        # one-hot to label
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        loss2 = config["alpha"] * (u.abs() - 1).pow(3).abs().mean()
        return loss1 + loss2

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


class DHNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DHNLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float()
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float()

        if config["GPU"]:
            self.U = self.U.cuda()
            self.Y = self.Y.cuda()

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        quantization_loss = config["alpha"] * (u.abs() - 1).abs().mean()

        return likelihood_loss + quantization_loss

class IDHNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(IDHNLoss, self).__init__()
        self.q = bit
        self.U = torch.zeros(config["num_train"], bit).float()
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float()

        if config["GPU"]:
            self.U = self.U.cuda()
            self.Y = self.Y.cuda()

    def forward(self, u, y, ind, config):
        u = u / (u.abs() + 1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = y @ self.Y.t()
        norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ self.Y.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        s = s / (norm + 0.00001)

        M = (s > 0.99).float() + (s < 0.01).float()

        inner_product = config["alpha1"] * u @ self.U.t()

        log_loss = torch.log(1 + torch.exp(-inner_product.abs())) + inner_product.clamp(min=0) - s * inner_product

        mse_loss = (inner_product + self.q - 2 * s * self.q).pow(2)

        loss1 = (M * log_loss + config["gamma1"] * (1 - M) * mse_loss).mean()
        loss2 = config["lambda1"] * (u.abs() - 1).abs().mean()

        return loss1 + loss2

class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).cuda()
        self.multi_label_random_center = torch.randint(2, (bit,)).float().cuda()
        self.criterion = torch.nn.BCELoss().cuda()

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + config["lambda"] * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

class DTSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y, ind, config):

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - config["alpha"]).clamp(min=-100,
                                                                                                             max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = config["lambda"] * (u - u.sign()).pow(2).mean()

        return loss1 + loss2

def train_val(config, bit):
    config["m"] = 2 * bit


    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)

    config["num_train"] = num_train
    net = config["net"](bit)



    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    U = torch.zeros(num_train, bit).float()
    L = torch.zeros(num_train, config["n_class"]).float()

    if config["GPU"]:
        U = U.cuda()
        L = L.cuda()

    # criterion = DPSHLoss(config, bit)
    # criterion1 = DSHLoss(config, bit)
    criterion = GreedyHashLoss(config, bit)
    # criterion3 = IDHNLoss(config, bit)
    criterion1 = DHNLoss(config, bit)
    criterion2 = IDHNLoss(config, bit)
    criterionCSQ = CSQLoss(config, bit)
    criterion4 = DTSHLoss(config, bit)

    Best_mAP = 0
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="") #

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:

            # print("000000000000000000000000000000000000000000000000000")  #test_loader     train_loader
            # print(image.size())
            # print(label.size())
            if config["GPU"]:
                image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()
            b = net(image)
            # print(b.size())
            # print("1111111111111111111111111111111111111111111111111111111")
            U[ind,: ] = b.data
            L[ind, :] = label.float()
            # b = torch.Tensor(b)
            # U = torch.Tensor(U)

            # loss = pairwise_loss(b, b, label, label)
            # loss = calc_loss(b, b, label.float(), label.float(), config)+calc_loss1(b, U, label.float(), L, config)

            # loss = criterion(b, label.float(), ind, config)+criterion1(b, label.float(), ind, config)+criterion2(b, label.float(), ind, config)+criterion3(b, label.float(), ind, config)
            loss = criterion(b, label.float(), ind, config)+ criterion1(b, label.float(), ind, config)+criterion2(b, label.float(), ind, config)+criterionCSQ(b, label.float(), ind, config)+criterion4(b, label.float(), ind, config)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        logger.info(str(bit)+'train_loss: {:.4f}'.format(train_loss))
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        # f.writelines(train_loss)

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])

            # print("calculating dataset binary code.......")
            # trn_binary, trn_label = compute_result(train_loader, net, usegpu=config["GPU"])
            trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])

            # print("calculating map.......")

            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            print(
                "%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (config["info"], epoch + 1, bit, config["dataset"], mAP))
            print(config)
            if mAP > Best_mAP:
                Best_mAP = mAP
                # torch.save(net.state_dict(), './checkpoint/BestNet.pth')		
    print("bit:%d,Best MAP:%.3f" % (bit, Best_mAP))
    logger.info(str(bit)+'_map: {:.4f}'.format(Best_mAP))


    if bit == 16: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')

        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

    
    if bit == 24: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')

        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')


    if bit == 32: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'tst_binary.txt',tst_binary,fmt='%2.1f')
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'tst_label.txt',tst_label,fmt='%2.1f')
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'trn_binary.txt',trn_binary,fmt='%2.1f')
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'trn_label.txt',trn_label,fmt='%2.1f')

    if bit == 36: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

    if bit == 48: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

    
    if bit == 64: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')


    if bit == 96: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

    
    if bit == 128: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

    if bit == 256: 
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result/'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
        tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])
        trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])
        P1, R1 = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p1.txt',P1,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r1.txt',R1,fmt='%3.5f')
        P, R = pr_curve1(tst_binary,trn_binary,tst_label,trn_label)
        # p_topK = p_top1(tst_binary,trn_binary,tst_label,trn_label)
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

if __name__ == "__main__":
    logger.add('logs/GreedyHash-DHN-IDHN-CSQ-DTSHrs{time}.log')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
