import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from util.util import gen_classes_per_node, gen_data_split, dirichlet_split_noniid
import copy
import argparse


def freeze_grad(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 13:34
@Author: KI
@File: ScaffoldOptimizer.py
@Motto: Hungry And Humble
"""
from torch.optim import Optimizer

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']


class Sever():
    def __init__(self, client_list, align_dataset, test_dataset,
                 device='cuda:1',
                 net='resnet50',
                 class_num=6,
                 client_num=3,
                 num_epochs=50,
                 client_align=True,
                 global_align=False,
                 align_batch_size=64,
                 learning_rate=0.01,
                 scaffold=False,
                 fedbn=False,
                 fednova=False):
        # 定义设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if net == 'resnet50':
            # Load ResNet-50 model
            self.global_model = models.resnet50(pretrained=True)
            self.global_model.fc = nn.Linear(self.global_model.fc.in_features, class_num)
        elif net == 'alexnet':
            self.global_model = models.alexnet(pretrained=True)
            self.global_model.classifier[6] = torch.nn.Linear(self.global_model.classifier[6].in_features, class_num)
        elif net == 'googlenet':
            self.global_model = models.googlenet(pretrained=True)
            self.global_model.fc = nn.Linear(self.global_model.fc.in_features, class_num)
        else:
            assert 'cant find net'
        self.global_model.to(self.device)
        self.client_num = client_num
        self.num_epochs = num_epochs
        self.client_list = client_list
        self.client_align = client_align
        self.global_align = global_align
        self.align_loader = DataLoader(align_dataset, batch_size=align_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=align_batch_size, shuffle=True)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.accuracy_list = []
        self.scaffold = scaffold
        if self.scaffold:
            self.cg = copy.deepcopy(self.global_model)
            for param in self.cg.parameters():
                param.data.fill_(0.0)
            self.eta = 1
        self.fedbn = fedbn
        self.fednova = fednova
        if self.fednova:
            self.n_data = 0
            self.n_data_list = []
            self.train_loss_list = []
            self.coeff_list = []
            self.norm_grad = []
        self.local_models = []

    def scaffold_iterate(self):
        dys = []
        dcs = []
        for i in range(self.client_num):
            dy, dc = self.client_list[i].scaffold_update(self.global_model, self.cg)
            dys.append(dy)
            dcs.append(dc)
        return dys, dcs

    def iterate(self):
        self.local_models = []
        if self.fednova:
            self.n_data = 0
            self.coeff_list = []
            self.norm_grad_list = []
            self.n_data_list = []
        for i in range(self.client_num):
            if self.fednova:
                local_model, n_data, coeff, norm_grad = self.client_list[i].update(self.global_model)
                self.n_data += n_data
                self.coeff_list.append(coeff)
                self.norm_grad_list.append(norm_grad)
                self.n_data_list.append(n_data)
            else:
                local_model = self.client_list[i].update(self.global_model)
            self.local_models.append(local_model)

    def align_train(self, model, optimizer):
        model.train()
        total_loss = 0
        for data, target in self.align_loader:
            data = data.to(self.device)  # 将输入放到设备上
            target = target.to(self.device)  # 将标签放到设备上
            optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss {total_loss / len(self.align_loader)}')

    #客户端client-drift 对齐
    def client_alignment(self):
        for client_id in range(self.client_num):
            # 用每个客户端的关键特征进行微调
            optimizer = torch.optim.SGD(self.local_models[client_id].parameters(), lr=self.learning_rate, momentum=0.9)  # 随机梯度下降优化器
            self.align_train(self.local_models[client_id], optimizer)

    #全局模型client-drift对齐，该操作慎用
    def global_alignment(self):
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate, momentum=0.9)  # 随机梯度下降优化器
        train_loss = self.align_train(self.global_model, optimizer)
        print(f'global_alignment Loss {train_loss}')

    def aggregate(self):
        #################################################################
        #fednova关键操作
        if self.fednova:
            coeff = 0.0
            global_state_dict = self.global_model.state_dict()
            nova_model_state = copy.deepcopy(global_state_dict)
            for i in range(len(self.coeff_list)):
                coeff = coeff + self.coeff_list[i] * self.n_data_list[i] / self.n_data
                for key in nova_model_state:
                    if 'num_batches_tracked' in key:
                        continue
                    if i == 0:
                        nova_model_state[key] = self.norm_grad_list[i][key] * self.n_data_list[i] / self.n_data
                    else:
                        nova_model_state[key] = nova_model_state[key] + self.norm_grad_list[i][key] * self.n_data_list[i] / self.n_data
            for key in global_state_dict.keys():
                if 'num_batches_tracked' in key:
                    continue
                global_state_dict[key] -= coeff * nova_model_state[key]
            self.global_model.load_state_dict(global_state_dict)
        #################################################################
        else:
            avg_state_dict = self.model_average(self.local_models)
            #################################################################
            #fedbn关键操作
            if self.fedbn:
                global_state_dict = self.global_model.state_dict()
                for key in global_state_dict.keys():
                    if 'norm' not in key:
                        global_state_dict[key] = avg_state_dict[key]
                self.global_model.load_state_dict(global_state_dict)
            #################################################################
            else:
                self.global_model.load_state_dict(avg_state_dict)

    def model_average(self, models):
        with torch.no_grad():
            avg_state_dict = copy.deepcopy(self.global_model.state_dict())
            for key in avg_state_dict.keys():
                if 'num_batches_tracked' in key:
                    continue
                avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)
        return avg_state_dict

    def model_sum(self, models):
        sum_state_dict = models[0].state_dict()
        for key in sum_state_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            sum_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).sum(0)
        return sum_state_dict

    ##########################################################
    #scaffold server端修改部分
    def scaffold_aggregate(self, dys, dcs):
        avg_state_dict = self.model_average(dys)
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            global_state_dict[key] += self.eta*avg_state_dict[key]
        self.global_model.load_state_dict(global_state_dict)
        sum_state_dict = self.model_sum(dcs)
        cg_state_dict = self.cg.state_dict()
        for key in cg_state_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            cg_state_dict[key] += sum_state_dict[key]/self.client_num
        self.cg.load_state_dict(cg_state_dict)
    ##########################################################

    def test(self):
        self.global_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return correct / len(self.test_loader.dataset)

    ##########################################################
    #scaffold server端修改部分
    def scaffold_run(self):
        for i in range(self.num_epochs):
            dys, dcs = self.scaffold_iterate()
            self.scaffold_aggregate(dys, dcs)
            acc = self.test()
            print(f'Test Accuracy: {acc}')
            self.accuracy_list.append(acc)
    ##########################################################
    def run(self):
        for i in range(self.num_epochs):
            self.iterate()
            if self.client_align:
                self.client_alignment()
            self.aggregate()
            if self.global_align:
                self.global_alignment()
            acc = self.test()
            print(f'Test Accuracy: {acc}')
            self.accuracy_list.append(acc)

    def draw(self):
        import matplotlib.pyplot as plt
        print(self.accuracy_list)
        plt.figure(figsize=(4, 4))
        plt.plot(self.accuracy_list, label='accuracy')
        plt.legend()
        plt.title('training accuracy')
        plt.savefig('../result/acc.png')
        plt.show()
        # Save the trained model
        torch.save(self.global_model.state_dict(), '../result/cifar10_resnet50.pth')
        print('Model saved')


class Client():

    def __init__(self, client_loader, client_id,
                 device='cuda:1',
                 net='resnet50',
                 class_num=6,
                 learning_rate=0.01,
                 fedprox=False,
                 scaffold=False,
                 fedbn=False,
                 fednova=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.client_loader = client_loader
        if net == 'resnet50':
            # Load ResNet-50 model
            self.local_model = models.resnet50(pretrained=True)
            self.local_model.fc = nn.Linear(self.local_model.fc.in_features, class_num)
        elif net == 'alexnet':
            self.local_model = models.alexnet(pretrained=True)
            self.local_model.classifier[6] = torch.nn.Linear(self.local_model.classifier[6].in_features, class_num)
        elif net == 'googlenet':
            # Load ResNet-50 model
            self.local_model = models.googlenet(pretrained=True)
            self.local_model.fc = nn.Linear(self.local_model.fc.in_features, class_num)
        else:
            assert 'cant find net'
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.local_model.to(self.device)  # 将模型放到设备上
        self.client_id = client_id
        self.fedprox = fedprox
        self.c = None
        self.learning_rate = learning_rate
        self.fedbn = fedbn
        self.fednova = fednova
        if self.fednova:
            self.rho = 0.9
            self._momentum = self.rho
            self.tau = len(self.client_loader)
            self.n_data = len(self.client_loader.dataset)

    ##########################################################
    # scaffold关键功能
    def scaffold_train(self, cg):
        optimizer = ScaffoldOptimizer(self.local_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.local_model.train()
        src_model = copy.deepcopy(self.local_model)
        src_model= freeze_grad(src_model)
        cg = freeze_grad(cg)
        if self.c is None: self.c = copy.deepcopy(cg)
        self.c = freeze_grad(self.c)
        total_loss = 0
        num_steps = 0
        for data, target in self.client_loader:
            num_steps += 1
            data = data.to(self.device)  # 将输入放到设备上
            target = target.to(self.device)  # 将标签放到设备上
            optimizer.zero_grad()
            output = self.local_model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step(cg.parameters(), self.c.parameters())
            total_loss += loss.item()
        dy_state_dict = self.local_model.state_dict()
        src_state_dict = src_model.state_dict()
        for key in dy_state_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            dy_state_dict[key] -= src_state_dict[key]
        self.local_model.load_state_dict(dy_state_dict)
        dc = copy.deepcopy(self.local_model)
        dc_state_dict = dc.state_dict()
        cg_state_dict = cg.state_dict()
        for key in dc_state_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            dc_state_dict[key] = -dc_state_dict[key]/(num_steps * self.learning_rate) - cg_state_dict[key]
        dc.load_state_dict(dc_state_dict)
        c_state_dict = self.c.state_dict()
        for key in c_state_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            c_state_dict[key] += dc_state_dict[key]
        self.c.load_state_dict(c_state_dict)
        return total_loss / len(self.client_loader), self.local_model, dc
    ##########################################################

    def local_train(self, global_model):
        total_loss = 0
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate, momentum=0.9)  # 随机梯度下降优化器
        self.local_model.train()
        for data, target in self.client_loader:
            data = data.to(self.device)  # 将输入放到设备上
            target = target.to(self.device)  # 将标签放到设备上
            optimizer.zero_grad()
            output = self.local_model(data)
            ######################################################
            if self.fedprox: #fedprox关键功能
                # fedprox近端项
                proximal_term = 0.0
                mu = 0.01
                for w, w_t in zip(self.local_model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = self.criterion(output, target) + (mu / 2) * proximal_term
            ######################################################
            else:
                loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.client_loader)

    def scaffold_update(self, global_model, cg):
        self.local_model.load_state_dict(global_model.state_dict())
        train_loss, dy, dc = self.scaffold_train(cg)
        print(f'Client {self.client_id}, Loss {train_loss}')
        return dy, dc

    def update(self, global_model):
        #################################################
        #fedbn关键操作
        if self.fedbn:
            local_state_dict = self.local_model.state_dict()
            global_state_dict = global_model.state_dict()
            for key in local_state_dict.keys():
                if 'norm' not in key:
                    local_state_dict[key] = global_state_dict[key]
            self.local_model.load_state_dict(local_state_dict)
        ##################################################
        else:
            self.local_model.load_state_dict(global_model.state_dict())
        train_loss = self.local_train(global_model)
        print(f'Client {self.client_id}, Loss {train_loss}')
        ##################################################
        #fednova关键操作
        if self.fednova:
            coeff = (self.tau - self.rho * (1 - pow(self.rho, self.tau)) / (1 - self.rho)) / (1 - self.rho)
            state_dict = self.local_model.state_dict()
            norm_grad = global_model.state_dict()
            for key in norm_grad.keys():
                norm_grad[key] = torch.div(norm_grad[key] - state_dict[key], coeff)
            return self.local_model, self.n_data, coeff, norm_grad
        ##################################################
        return self.local_model


if __name__ == '__main__':
    def args_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--net', type=str, default='resnet50', help='classiffier network')
        parser.add_argument('--class_num', type=int, default=6, help='number of classes')
        parser.add_argument('--client_num', type=int, default=2, help='number of clients')
        parser.add_argument('--client_class_num', type=int, default=3, help='number of clients class')
        parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--align_datadir', type=str, default='/home/chase/shy/FedDiffG/data/NEU-CLS/DGDA', help='datadir of align dataset')
        parser.add_argument('--train_datadir', type=str, default='/home/chase/shy/FedDiffG/data/NEU-CLS/train', help='datadir of train dataset')
        parser.add_argument('--test_datadir', type=str, default='/home/chase/shy/FedDiffG/data/NEU-CLS/test', help='datadir of test dataset')
        parser.add_argument('--align_batch_size', type=int, default=32, help='align batch_size')
        parser.add_argument('--local_batch_size', type=int, default=32, help='local clients batch_size')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
        parser.add_argument('--client_align', type=bool, default=True, help='align process')
        parser.add_argument('--global_align', type=bool, default=True, help='align process')
        parser.add_argument('--fedprox', type=bool, default=False, help='add fedprox proximal_term')
        parser.add_argument('--scaffold', type=bool, default=False, help='add scaffold')
        parser.add_argument('--fedbn', type=bool, default=False, help='add fedbn')
        parser.add_argument('--fednova', type=bool, default=False, help='add fednova')
        parser.add_argument('--dirichlet', type=bool, default=False, help='use dirichlet to split no-iid dataset')
        parser.add_argument('--distribution_dir', type=str, default='../result/distribution_NEU-CLS_client2.npy', help='use the distribution as file')
        parser.add_argument('--save_distribution_dir', type=str, default='../result/distribution_NEU-CLS_client2.npy', help='save distribution')
        parser.add_argument('--r', type=float, default=1.0, help='alpha of dirichlet')

        args = parser.parse_args()
        return args

    args = args_parser()
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.ImageFolder(args.train_datadir, transform)
    align_dataset = datasets.ImageFolder(args.align_datadir, transform)
    test_dataset = datasets.ImageFolder(args.test_datadir, transform)
    loader_params = {"batch_size": args.local_batch_size, "shuffle": True, "pin_memory": True, "num_workers": 0}
    #划分no-iid数据集
    if args.distribution_dir is not None:
        usr_subset_idx = np.load(args.distribution_dir, allow_pickle=True)
    else:
        if args.dirichlet:
            usr_subset_idx = dirichlet_split_noniid(train_dataset.targets, alpha=args.r, n_classes=args.class_num, n_clients=args.client_num)
        else:
            cls_partitions = gen_classes_per_node(train_dataset, args.client_num, args.client_class_num)
            print(cls_partitions)
            #[[3, 0], [2, 5], [4, 1]] 0.75
            #[[2, 0], [4, 3], [1, 5]] 0.78
            #[[3, 5], [2, 4], [0, 1]] 0.9
            #[[3, 4], [5, 2], [1, 0]] 0.9
            #[[4, 2], [0, 3], [5, 1]] 0.8
            #[[3, 0, 5], [1, 4, 2]]
            #[[4, 5, 3], [1, 2, 0]] 0.96
            #[[2, 0, 5], [4, 3, 1]] 0.96
            #[[2, 1, 3], [5, 4, 0]] 0.92
            #[[0, 3, 5], [1, 2, 4]]
            #[[1, 3, 0], [5, 4, 2]]0.97
            #[[0, 3, 4], [2, 5, 1]]
            #[[4, 5, 1], [0, 2, 3]]
            #[[5, 0, 1], [2, 4, 3]]
            #[[4, 0, 2], [1, 3, 5]]
            usr_subset_idx = gen_data_split(train_dataset, args.client_num, cls_partitions)
        if args.save_distribution_dir is not None:
            np.save(args.save_distribution_dir, usr_subset_idx)
    # create subsets for each client
    subsets = list(map(lambda x: torch.utils.data.Subset(train_dataset, x), usr_subset_idx))
    # create dataloaders from subsets
    train_dataloaders = list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets))

    #初始化客户端
    client_list = []
    for i in range(args.client_num):
        client = Client(train_dataloaders[i], i,
                        device=args.device,
                        net=args.net,
                        class_num=args.class_num,
                        learning_rate=args.learning_rate,
                        fedprox=args.fedprox,
                        scaffold=args.scaffold,
                        fedbn=args.fedbn,
                        fednova=args.fednova)
        client_list.append(client)

    #初始化server端
    sever = Sever(client_list, align_dataset, test_dataset,
                  device=args.device,
                  net=args.net,
                  class_num=args.class_num,
                  client_num=args.client_num,
                  num_epochs=args.num_epochs,
                  client_align=args.client_align,
                  global_align=args.global_align,
                  align_batch_size=args.align_batch_size,
                  learning_rate=args.learning_rate,
                  scaffold=args.scaffold,
                  fedbn=args.fedbn,
                  fednova=args.fednova)
    if args.scaffold:
        sever.scaffold_run()
    else:
        sever.run()
    sever.draw()



