
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from multilabeldata import MultiLabelDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from PIL import Image
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler import GradualWarmupScheduler
from torchvision import datasets, transforms

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    # dataset = CIFAR10(
    #     root='./CIFAR10', train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    # dataloader = DataLoader(
    #     dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(modelConfig["img_size"], modelConfig["img_size"]), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = MultiLabelDataset(modelConfig["data_dir"], data_transforms)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    # model setup
    net_model = UNet(T=modelConfig["T"],  num_labels=modelConfig["num_labels"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, (cls_label, shape_label) in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                cls_label = cls_label.to(device)
                shape_label = shape_label.to(device)
                loss = trainer(x_0, cls_label, shape_label).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()

    if not os.path.exists(modelConfig["save_weight_dir"]):
        os.makedirs(modelConfig["save_weight_dir"])
    torch.save(net_model.state_dict(), os.path.join(
        modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], num_labels=modelConfig['num_labels'], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution
        savePath = modelConfig["save_weight_dir"]
        classes = ['bianping', 'damo', 'duoge', 'fangda', 'shikuai']
        with torch.no_grad():
            for c in range(len(classes)):
                for m in range(1):
                    labelList = []
                    for i in range(0, modelConfig["batch_size"]):
                        labelList.append(torch.ones(size=[1]).long() * c)
                    labels = torch.cat(labelList, dim=0).long().to(device)
                    # Sampled from standard normal distribution
                    noisyImage = torch.randn(
                        size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]],
                        device=device)
                    saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
                    save_image(saveNoisy, os.path.join(
                        modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
                    sampledImgs = sampler(noisyImage, labels)
                    sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                    if not os.path.exists(savePath + '/' + classes[c]):
                        os.makedirs(savePath + '/' + classes[c])
                    for i in range(len(sampledImgs)):
                        ndarr = sampledImgs[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                     torch.uint8).numpy()
                        im = Image.fromarray(ndarr)
                        im.save(savePath + '/' + classes[c] + '/' + str(m * modelConfig["batch_size"] + i) + '.jpg',
                                format=None)
        # noisyImage = torch.randn(
        #     size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        # sampledImgs = sampler(noisyImage)
        # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])