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

# Define the number of clients
num_clients = 3
client_class_num = 2
num_epochs = 50
batch_size = 64
learning_rate = 0.01
class_num = 6
client_alignment_status = False
global_alignment_status = False
fedprox = False
scaffold = True


# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(size=(64, 64), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_imgPath = '/home/chase/shy/DDPM4Industry/data/MT/train'
train_imgPath2 = '/home/chase/shy/DDPM4Industry/data/MT/GDDA'
test_imgPath = '/home/chase/shy/DDPM4Industry/data/MT/test'
train_dataset = datasets.ImageFolder(train_imgPath, transform)
train_dataset2 = datasets.ImageFolder(train_imgPath2, transform)
test_dataset = datasets.ImageFolder(test_imgPath, transform)



# Split dataset into clients
def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list

def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    # assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes + 1
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx

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
def train(client_loader, model, criterion, optimizer, global_model):
    model.train()
    total_loss = 0
    for data, target in client_loader:
        data = data.to(device) # 将输入放到设备上
        target = target.to(device) # 将标签放到设备上
        optimizer.zero_grad()
        output = model(data)
        # fedprox近端项
        proximal_term = 0.0
        if fedprox:
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)
        if proximal_term != 0:
            loss = criterion(output, target) + (0.01 / 2) * proximal_term
        else:
            loss = criterion(output, target) + (0.01 / 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Loss {loss.item()}')
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

def client_alignment(models):
    alignment_models = []
    for client_id in range(num_clients):
        # 用每个客户端的关键特征进行微调
        train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
        # 定义损失函数和优化器
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)  # 随机梯度下降优化器
        client_model = models[client_id]
        train_loss = train(train_loader2, client_model, criterion, optimizer)
        alignment_models.append(client_model)
        print(f'client_alignment Client {client_id}, Loss {train_loss}')
    return alignment_models

def global_alignment(model):
    # 用每个客户端的关键特征进行微调
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)  # 随机梯度下降优化器
    train_loss = train(train_loader2, model, criterion, optimizer)
    print(f'global_alignment Client {client_id}, Loss {train_loss}')
    return model

# FedAvg function
def fed_avg(models):
    if client_alignment_status:
        models = client_alignment(models)
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
        train_loss = train(client_loader, local_model, criterion, optimizer, global_model)
        local_models.append(local_model)
        print(f'Client {client_id}, Epoch {epoch}, Loss {train_loss}')
        loss_list.append(train_loss)
    
    # Perform FedAvg
    local_models = fed_avg(local_models)
    # Load the global model state dict
    global_model.load_state_dict(local_models[0].state_dict())
    global_model.to(device)
    if global_alignment_status:
        global_model = global_alignment(global_model)
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
torch.save(global_model.state_dict(), 'cifar10_resnet50.pth')
print('Model saved as cifar10_resnet50.pth')