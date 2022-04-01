from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import json
import numpy as np
import glob
import random
from tqdm import tqdm

class MelonDataset(Dataset):
    def __init__(self, songlist, songmesta):
        self.songlist = songlist
        self.songmeta = songmesta
        self.len = len(songlist)

    def __getitem__(self, idx):
        song_arr = np.load(self.songlist[idx])
        song_arr = song_arr[:,:577].reshape(1, 48, 577)
        song_idx = int(self.songlist[idx].split('/')[-1].rstrip('.npy'))
        label = self.songmeta[song_idx]['song_gn_gnr_basket']
        if len(label) >= 1:
            label = (int(random.choice(label).replace('GN', '')) // 100) - 1
        else:
            print('error')
        if label > 30:
            label = 9
        return song_arr, label

    def __len__(self):
        return self.len



class GenreClassification(nn.Module):
    def __init__(self):
        super(GenreClassification, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 1)
        self.mp1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.mp2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(16)
        self.mp3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(16)
        self.mp4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1120, 516) # input features, output features
        self.dp1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(516, 128)
        self.dp2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 30)
        self.sf = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.bn1(self.mp1(F.relu(self.conv2(F.relu(self.conv1(x)))))) # (32, 22, 936)
        x = self.bn2(self.mp2(F.relu(self.conv4(F.relu(self.conv3(x)))))) # (16, 11, 468)
        x = self.bn3(self.mp3(F.relu(self.conv6(F.relu(self.conv5(x)))))) # (16, 5, 234)
        x = self.bn4(self.mp4(F.relu(self.conv8(F.relu(self.conv7(x)))))) # (16, 2, 117) (16, 2, 35)
        x = F.relu(self.fc1(self.dp1(x.view(-1, 1120))))
        x = F.relu(self.fc2(self.dp2(x)))
        x = self.fc3(x)
        return self.sf(x)


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))

file_dir = './arena_mel/'
songlist = glob.glob(file_dir + '*/*.npy')
songmeta = './arena_data/song_meta.json'
with open(songmeta, encoding="utf-8") as f:
    songmeta = json.load(f)

for i in songmeta:
    if len(i['song_gn_gnr_basket']) == 0:
        songlist.pop(songlist.index(file_dir + str(i['id']//1000) + '/' + str(i['id']) + '.npy'))
        print(file_dir + str(i['id']//1000) + '/' + str(i['id']) + '.npy')
random.seed(100)
random.shuffle(songlist)

trainlist = songlist[:int(len(songlist)*0.7)]
vallist = songlist[int(len(songlist)*0.7):]

train_dataset = MelonDataset(trainlist, songmeta)
val_dataset = MelonDataset(vallist, songmeta)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128)
val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=128)


model = GenreClassification()
model.apply(weight_init)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

trloss, teloss, teacc = [], [], []


def plotdata(trl, tel, tea):
    xlist = range(len(trl))
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(xlist, trl, 'r-', label='train loss')
    plt.plot(xlist, tel, 'b-', label='validation loss')
    plt.ylabel('loss value')
    plt.title('loss graph')
    plt.legend(loc=1)

    ax2 = plt.subplot(2, 1, 2)
    plt.plot(xlist, tea, 'b-', label='validation acc')
    #plt.ylim(0, 100)
    #plt.xlim(0, 100)
    plt.yticks(range(0,101,10))
    plt.grid(True)
    plt.ylabel('acc(%)')
    plt.title('acc graph')
    plt.legend(loc=1)

    plt.tight_layout()

    plt.savefig('batchNorWithxavier.png', dpi=300)
    plt.close()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            trloss.append(loss.item())
            acc = test()
            plotdata(trloss, teloss, teacc)
    return acc

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(val_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    teloss.append(test_loss)
    teacc.append(100. * correct / len(val_loader.dataset))
    return correct


def save_checkpoint(state, filename='model_adam.pth.tar'):
    torch.save(state, filename)


acc_ = 0
for epoch in range(1, 201):
    accurancy = train(epoch)
    if acc_ < accurancy:
        acc_ = accurancy
        ep = epoch
save_checkpoint(ep)
print('model saved at %d epoch'%ep)