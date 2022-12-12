import torch 
import pickle
from model import Net
import os
from PIL import Image
import numpy as np

model = Net()
model.load_state_dict(torch.load('export_model.pth'))
model = model.eval()

with open('../data/label/price_normalize.pickle', 'rb') as f:
    prices_normalize = pickle.load(f)
    
label_mean_std = prices_normalize['mean_std']
mean = label_mean_std[0]
std = label_mean_std[1]

for input_fig in os.listdir('../data/predict'):
    image_path = "../data/predict/{}".format(input_fig)
    im = Image.open(image_path)

    resize_h = 400
    resize_w = 400
    im = im.resize([resize_h, resize_w])
    im = np.array(im).astype(np.float32).transpose(2,1,0)
    im = torch.from_numpy(im.astype(np.float32)).clone()
    with torch.no_grad():      
        output = model(im)
        output = output*std + mean
    print(output)

