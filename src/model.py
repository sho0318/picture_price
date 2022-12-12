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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)


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
        image = image / 255

        label = self.train_df["label"].apply(lambda x: float(x)).to_list()[index]

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
        self.fc1 = nn.Linear(in_features=128*100*100, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 128*100*100)
        x = self.fc1(x)
        x = self.dropout(x)

        x = F.relu(x)
        x = self.tanh(x)
        x = self.fc2(x)
        
        return x

def train(net, optimizer, criterion, dataloaders_dict):
    EPOCHS = 5
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, EPOCHS))

        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    outputs = net(inputs)
                    print(outputs)
                    print(phase)

                    labels = np.array(labels).reshape([len(labels),1])
                    labels = torch.from_numpy(labels.astype(np.float32)).clone()
                    labels = labels.to(device)
                    # print(labels)

                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        print('-------------------')



def dataload():
    with open('../data/label/price_normalize.pickle', 'rb') as f:
        prices_normalize = pickle.load(f)
    
    label_mean_std = prices_normalize['mean_std']
    prices = prices_normalize['label']

    DIR = "../data/fig"
    data = []
    for num in os.listdir(DIR):
        data_num = int(num.split('.')[0])
        data.append([f'{DIR}/{num}', num, prices[data_num]])

    df = pd.DataFrame(data, columns=['path', 'filename', 'label'])

    BATCH_SIZE = 64
    SIZE = 400
    TRAINDATA_RATE = 0.8

    image_dataset = MyDatasete(df, (SIZE, SIZE))

    train_dataset, valid_dataset = torch.utils.data.random_split(image_dataset, [int(len(image_dataset)*TRAINDATA_RATE), len(image_dataset)-int(len(image_dataset)*TRAINDATA_RATE)])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2,
        drop_last = True,
        pin_memory = True
    )

    # valid_dataloader = DataLoader(
    #     valid_dataset,
    #     batch_size = BATCH_SIZE,
    #     shuffle = True,
    #     num_workers = 2,
    #     drop_last = True,
    #     pin_memory = True
    # )

    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


    dataloaders_dict = {'train': train_dataloader, 'valid': valid_dataloader}

    net = Net()
    net = net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    nll_loss = nn.NLLLoss()

    train(net, optimizer, criterion, dataloaders_dict)
    torch.save(net.state_dict(), "export_model.pth")



if __name__ == "__main__":
    dataload()