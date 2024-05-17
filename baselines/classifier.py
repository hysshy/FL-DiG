# 导入必要的库
import torch
import torchvision
from torchvision import datasets, transforms
# 定义超参数
batch_size = 64 # 批次大小
num_classes = 6 # 分类数目，根据数据集修改
num_epochs = 50 # 训练轮数
learning_rate = 0.01 # 学习率

# 定义数据变换

transform_train = transforms.Compose([
    # transforms.RandomCrop(64, padding=4), # 随机裁剪
    transforms.Resize(size=(64, 64), interpolation=3),
    # transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
])

transform_test = transforms.Compose([
    transforms.Resize(size=(64, 64), interpolation=3),
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
])
train_imgPath = '/home/chase/shy/FedDiffG/data/NEU-CLS/F-train'
test_imgPath = '/home/chase/shy/FedDiffG/data/NEU-CLS/test'
# 加载数据集
trainset = datasets.ImageFolder(train_imgPath, transform_train)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) # 训练集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2) # 训练加载器

testset = datasets.ImageFolder(test_imgPath, transform_test)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) # 测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2) # 测试加载器

# 定义分类标签
# classes = ['biotite', 'bornite', 'chrysocolla', 'malachite', 'muscovite', 'pyrite', 'quartz']

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
model = torchvision.models.resnet50(pretrained=True) # 加载预训练的ResNet18模型
model.fc = torch.nn.Linear(model.fc.in_features, num_classes) # 修改全连接层的输出
# model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes) # 修改全连接层的输出
# model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes) # 修改全连接层的输出
model.to(device) # 将模型放到设备上

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # 随机梯度下降优化器
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 定义训练函数
def train(epoch):
    model.train() # 切换到训练模式
    running_loss = 0.0 # 累计损失
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data # 获取输入和标签
        inputs = inputs.to(device) # 将输入放到设备上
        labels = labels.to(device) # 将标签放到设备上
        optimizer.zero_grad() # 清零梯度
        outputs = model(inputs) # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        running_loss += loss.item() # 累加损失
        if i % 20 == 0: # 每200个批次打印一次损失
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    # scheduler.step()
# 定义测试函数
def test():
    model.eval() # 切换到评估模式
    correct = 0 # 正确预测的数量
    total = 0 # 总共的数量
    with torch.no_grad(): # 关闭梯度计算
        for data in testloader:
            images, labels = data # 获取输入和标签
            images = images.to(device) # 将输入放到设备上
            labels = labels.to(device) # 将标签放到设备上
            outputs = model(images) # 前向传播
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果
            total += labels.size(0) # 累加总数
            correct += (predicted == labels).sum().item() # 累加正确数
    print('Accuracy of the network on the test images: %d %%' % (100* correct / total)) # 打印准确率

# 开始训练和测试
for epoch in range(num_epochs):
    train(epoch) # 训练
    print(epoch)
    test() # 测试