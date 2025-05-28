import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchsummary import summary
import torchutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torchutils.layer.LearnDepthwiseConv(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.CBAM1 = torchutils.layer.CBAM(channel=32)
        self.layer1 = torchutils.layer.make_layer(block=torchutils.layer.ResidualBlock, in_channels=32, out_channels=64, blocks=2)
        self.CBAM2 = torchutils.layer.CBAM(channel=64)
        self.layer2 = torchutils.layer.make_layer(block=torchutils.layer.ResidualBlock, in_channels=64, out_channels=128, blocks=2, stride=2)
        self.CBAM3 = torchutils.layer.CBAM(channel=128)

        self.bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.gated = torchutils.activatefunc.GatedActivation(nn.ReLU())

        self.noise1 = torchutils.layer.LearnableRandomNoise(channels=32)
        self.noise2 = torchutils.layer.LearnableRandomNoise(channels=64)
        self.noise3 = torchutils.layer.LearnableRandomNoise(channels=128)
    def forward(self, x):
        # print()
        # print(x.shape)
        x = self.relu(self.bn(self.pool(self.conv1(x))))
        x = self.noise1(x)
        x = self.CBAM1(x)
        x = self.layer1(x)
        x = self.noise2(x)
        x = self.CBAM2(x)
        x = self.layer2(x)
        x = self.noise3(x)
        x = self.CBAM3(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.gated(self.dropout(self.fc1(x)))
        # print(x.shape)
        x = self.gated(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


transform_train = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    torchutils.transform.FourierReduct(device='cpu', coef=0.8),
    torchutils.transform.InverseColor(device='cpu'),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_full = datasets.MNIST(root='data', train=True, transform=transform_train, download=True)
test_dataset_full = datasets.MNIST(root='data', train=False, transform=transform_test, download=True)

subset_train_size = int(0.1 * len(train_dataset_full))
subset_test_size = int(0.1 * len(test_dataset_full))

subset_indices_train = random.sample(range(len(train_dataset_full)), subset_train_size)
subset_indices_test = random.sample(range(len(test_dataset_full)), subset_test_size)

train_dataset = torch.utils.data.Subset(train_dataset_full, subset_indices_train)
test_dataset = torch.utils.data.Subset(test_dataset_full, subset_indices_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNN()
# summary(model, (1, 28, 28))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

history_train_loss = []
history_val_loss = []
history_val_acc = []

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        #images = images.squeeze(1)
        images = images.to('cpu')
        labels = labels.to('cpu')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    running_loss += loss.item()
    history_train_loss.append(running_loss/len(train_loader))

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        running_loss_eval = 0.0
        for images, labels in tqdm(test_loader):
            images = images.to('cpu')
            labels = labels.to('cpu')
            outputs = model(images)
            eval_loss = criterion(outputs, labels)
            running_loss_eval += eval_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    history_val_loss.append(running_loss_eval/len(test_loader))
    history_val_acc.append(100 * correct / total)
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Loss: {running_loss_eval/len(test_loader)}, Val Acc: {100 * correct / total}%')


model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

plt.plot(history_train_loss, label='Train Loss')
plt.plot(history_val_loss, label='Validation Loss')
plt.plot(history_val_acc, label='Validation Accuracy')
plt.legend()
plt.show()