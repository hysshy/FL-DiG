import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import defaultdict
import random
import torch.optim.lr_scheduler as lr_scheduler
from util.util import get_num_classes_samples, gen_classes_per_node, gen_data_split
# Define the number of clients
num_clients = 2
client_class_num = 3
num_epochs = 80
batch_size = 64
learning_rate = 0.01
class_num = 6
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(size=(64, 64), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train_imgPath = '/home/chase/shy/DGDA/data/MT/train'
test_imgPath = '/home/chase/shy/DGDA/data/MT/test'
train_dataset = datasets.ImageFolder(train_imgPath, transform)
test_dataset = datasets.ImageFolder(test_imgPath, transform)

loader_params = {"batch_size": batch_size, "shuffle": True, "pin_memory": True, "num_workers": 0}
cls_partitions = gen_classes_per_node(train_dataset, num_clients, client_class_num)
usr_subset_idx = gen_data_split(train_dataset, num_clients, cls_partitions)
# create subsets for each client
subsets = list(map(lambda x: torch.utils.data.Subset(train_dataset, x), usr_subset_idx))
# create dataloaders from subsets
train_dataloaders = list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets))

# 定义设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Load ResNet-50 model
global_model = models.resnet50(pretrained=True)
global_model.fc = nn.Linear(global_model.fc.in_features, class_num)  # CIFAR-10 has 10 classes

# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

# Training function
def train(client_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, target in client_loader:
        data = data.to(device) # 将输入放到设备上
        target = target.to(device) # 将标签放到设备上
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(client_loader)

# Testing function
def test(test_loader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)

# FedAvg function
def fed_avg(models):
    global_state_dict = models[0].state_dict()
    for key in global_state_dict.keys():
        if 'num_batches_tracked' in key:
            continue
        global_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)
    for model in models:
        model.load_state_dict(global_state_dict)
    return models

def test_accuracy(global_model):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_accuracy = test(test_loader, global_model, criterion)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    return test_accuracy

loss_list = []
accuracy_list = []
# Main training loop
for epoch in range(num_epochs):
    local_models = []
    for client_id in range(num_clients):
        client_loader = train_dataloaders[client_id]
        local_model = models.resnet50(pretrained=True)
        local_model.fc = nn.Linear(global_model.fc.in_features, class_num)
        local_model.load_state_dict(global_model.state_dict())
        local_model.to(device)  # 将模型放到设备上
        global_model.to(device)
        # 定义损失函数和优化器
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)  # 随机梯度下降优化器
        train_loss = train(client_loader, local_model, criterion, optimizer)
        local_models.append(local_model)
        print(f'Client {client_id}, Epoch {epoch}, Loss {train_loss}')
        loss_list.append(train_loss)
    
    # Perform FedAvg
    local_models = fed_avg(local_models)
    # Load the global model state dict
    global_model.load_state_dict(local_models[0].state_dict())
    # Test the global model
    accuracy_list.append(test_accuracy(global_model))

print(loss_list)
print(accuracy_list)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_list, label='loss')
plt.legend()
plt.title('training loss')

plt.subplot(1,2,2)
plt.plot(accuracy_list, label='accuracy')
plt.legend()
plt.title('training accuracy')
plt.savefig('fedavg-client_align-global_align.png-local_epoch25.png')
plt.show()
# Save the trained model
torch.save(global_model.state_dict(), '../cifar10_resnet50.pth')
print('Model saved as cifar10_resnet50.pth')