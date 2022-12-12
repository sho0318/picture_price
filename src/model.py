import os
import pickle

from PIL import Image 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class MyDatasete(Dataset):
    def __init__(self, train_df, input_size, phase='train', transform=None):
        super().__init__()
        self.train_df = train_df
        image_paths = train_df["path"].to_list()
        self.input_size = input_size
        self.len = len(image_paths)
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image_path = "../data/reshape_fig/{}.jpg".format(index)
        image = Image.open(image_path)

        image = np.array(image).astype(np.float32).transpose(2,1,0)
        label = self.train_df["label"].apply(lambda x: int(x)).to_list()[index]

        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128*800*800, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 128*800*800)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

def train(net, optimizer, criterion, dataloaders_dict):
    EPOCHS = 10
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, EPOCHS))
        print('-------------------')

        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



def dataload():
    with open('../data/label/price.pickle', 'rb') as f:
        prices = pickle.load(f)

    DIR = "../data/fig"
    data = []
    for num in os.listdir(DIR):
        data_num = int(num.split('.')[0])
        data.append([f'{DIR}/{num}', num, prices[data_num]])

    df = pd.DataFrame(data, columns=['path', 'filename', 'label'])


    BATCH_SIZE = 64
    SIZE = 400

    image_dataset = MyDatasete(df, (SIZE, SIZE))

    train_dataset, valid_dataset = torch.utils.data.random_split(image_dataset, [int(len(image_dataset)*0.7), len(image_dataset)-int(len(image_dataset)*0.7)])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2,
        drop_last = True,
        pin_memory = True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2,
        drop_last = True,
        pin_memory = True
    )

    dataloaders_dict = {'train': train_dataloader, 'valid': valid_dataloader}

    batch_iterator = iter(dataloaders_dict['train'])

    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    nll_loss = nn.NLLLoss()

    train(net, optimizer, criterion, dataloaders_dict)
    torch.save(net.state_dict(), "export_model.pth")



if __name__ == "__main__":
    dataload()