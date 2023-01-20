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

from data_load import preprocessing_data
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

FIG_SIZE = 300
TRAINDATA_RATE = 0.7

EPOCHS = 1
BATCH_SIZE = 128


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-dlf', '--DownloadFigData', type=bool,
                           default=False,
                           help='Whether to download figs from url')
    return argparser.parse_args()


class MyDatasete(Dataset):
    def __init__(self, train_df, input_size, phase='train', transform=None):
        super().__init__()
        self.train_df = train_df
        self.image_paths = train_df["paths"].to_list()
        self.input_size = input_size
        self.len = len(self.image_paths)
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if image.size != (300, 300):
            image = image.resize((300, 300), Image.LANCZOS)
            # Image.LANCZOSでいい感じに合わせてくれるらしい

        image = np.array(image).astype(np.float32).transpose(2,1,0)
        image = image / 255

        label = self.train_df["labels"].apply(lambda x: float(x)).to_list()[index]

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
        self.fc1 = nn.Linear(in_features=72*100*100, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

        self.Leaky = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout(0.1)
         
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        # x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        # x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 72*100*100)
        x = self.fc1(x)
        # x = self.dropout(x)

        x = self.fc2(x) 
        x = F.relu(x)
        
        
        return x


def train(net, optimizer, criterion, dataloaders_dict):
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, EPOCHS))

        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0.0
            for i, [inputs, labels] in enumerate(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # print(len(inputs))
                    outputs = net(inputs)
                    # print(len(outputs))
                    # print(phase)

                    labels = np.array(labels).reshape([len(labels),1])
                    labels = torch.from_numpy(labels.astype(np.float32)).clone()
                    labels = labels.to(device)
                    # print(len(labels))

                    loss = criterion(outputs, labels)
                    # if i % 10:
                    #     print(i, '回:', loss)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        print('-------------------')


def dataload(args):

    if args.DownloadFigData:
        with open('../data/this_is_gallery/df.pickle', 'rb') as f:
            df = pickle.load(f)

        df = preprocessing_data(df) 
        # df.columns = ['paths', 'labels'].  
        # paths:figのpath
        # labels:標準化後のprice
        with open('../data/this_is_gallery/preprocess_df.pickle', 'wb') as f:
            pickle.dump(df, f)
    else:
        with open('../data/this_is_gallery/preprocess_df.pickle', 'rb') as f:
            df = pickle.load(f)
    
    df = df.dropna()
    print(df)

    image_dataset = MyDatasete(df, (FIG_SIZE, FIG_SIZE))

    train_dataset, valid_dataset = torch.utils.data.random_split(image_dataset, 
                                                                 [int(len(image_dataset)*TRAINDATA_RATE), 
                                                                 len(image_dataset)-int(len(image_dataset)*TRAINDATA_RATE)])

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

    # valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    dataloaders_dict = {'train': train_dataloader, 'valid': valid_dataloader}

    return dataloaders_dict


def main(args):
    dataloaders_dict = dataload(args)

    net = Net()
    net = net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # nll_loss = nn.NLLLoss()

    train(net, optimizer, criterion, dataloaders_dict)
    torch.save(net.state_dict(), "export_model.pth")


if __name__ == "__main__":
    args = get_option()
    main(args)